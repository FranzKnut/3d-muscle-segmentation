import os
from deepvoxnet2.components.mirc import NiftiFileModality, Dataset, Case, Record
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.keras.models.unet_generalized import create_generalized_unet_model
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.components.transformers import MircInput, KerasModel, Put, Concat, RandomCropFullImage, Buffer, ArgMax

data_path = './data/BMD'
output_path = './prediction/BMD'
model_weights_path = './models/li_model_weights.h5'
case_ids = ['cc', 'early']

# create the mirc dataset
dataset = Dataset('LI', data_path)
for cid in case_ids:
    mirc_case = Case(cid)
    record = Record('0')
    record.add(NiftiFileModality('OP', os.path.join(data_path, 'OP_{}_r.nii.gz'.format(cid))))
    record.add(NiftiFileModality('IP', os.path.join(data_path, 'IP_{}_r.nii.gz'.format(cid))))
    record.add(NiftiFileModality('Mask', os.path.join(data_path, 'Mask_{}_r.nii.gz'.format(cid))))
    mirc_case.add(record)
    dataset.add(mirc_case)
mirc = Mirc(dataset)
sampler = MircSampler(mirc, shuffle=False)

output_dirs = [os.path.join(output_path, '{}'.format(i.case_id)) for i in sampler]
output_dirs.sort()
for p in output_dirs:
    if not os.path.isdir(p): os.makedirs(p)

segment_size = (163, 352, 55)
true_input_size = (189, 378, 81)

model = create_generalized_unet_model(
    number_input_features=2,
    subsample_factors_per_pathway=(
        (1, 1, 1),
        (3, 3, 3),
        (9, 9, 9),
        (27, 27, 27)),
    kernel_sizes_per_pathway=(
        (((3, 3, 1), (3, 3, 3)), ((3, 3, 3), (3, 3, 1))),
        (((3, 3, 1), (3, 3, 3)), ((3, 3, 3), (3, 3, 1))),
        (((3, 3, 1), (3, 3, 3)), ((3, 3, 3), (3, 3, 1))),
        (((3, 3, 3), (1, 1, 1)), ((1, 1, 1), (3, 3, 3)))),
    number_features_per_pathway=(
        ((15, 30), (30, 15)),
        ((40, 80), (80, 40)),
        ((80, 160), (160, 80)),
        ((160, 160), (160, 160))),
    output_size=segment_size,
    padding='same',
    upsampling='linear',
    l1_reg=0,
    l2_reg=0,
    batch_normalization=False,
    instance_normalization=True,
    activation_final_layer='softmax',
    number_features_common_pathway=(19,),
    dropout_common_pathway=(0,)
)

km_weights = model.load_weights(model_weights_path)
model_transformer = KerasModel(model)

# create the processing network
x_input_1 = MircInput(['OP'], output_shapes=[(1, None, None, None, 1)], name='OP_input', n=None)
x_input_2 = MircInput(['IP'], output_shapes=[(1, None, None, None, 1)], name='IP_input')
mask_input = MircInput(['Mask'], output_shapes=[(1, None, None, None, 1)], name='mask_input')

x_path_1, x_path_2 = RandomCropFullImage(mask_input, true_input_size, grid_size=segment_size, nonzero=True)(x_input_1, x_input_2)
x_path = Concat()([x_path_1, x_path_2])
x_dvn_val = model_transformer(x_path)
x_dvn_val = Buffer(buffer_size=None, drop_remainder=False)(x_dvn_val)
x_dvn_val = Put(x_input_1, keep_counts=True)(x_dvn_val)
x_dvn_val = ArgMax()(x_dvn_val)


dvn_model = DvnModel(
    outputs={
        'pred': [x_dvn_val]
    }
)

if len(output_dirs) != 0:
    dvn_model.predict('pred', sampler, output_dirs=output_dirs)
