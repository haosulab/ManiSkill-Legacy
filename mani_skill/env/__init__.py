from gym import register
import pathlib
from ..utils.misc import get_model_ids_from_yaml, get_raw_yaml

_this_file = pathlib.Path(__file__).resolve()

################################################################
# OpenCabinetDoor
################################################################

register(
    id='OpenCabinetDoor-v0',
    entry_point='mani_skill.env.open_cabinet_door_drawer:OpenCabinetDoorEnv',
)

cabinet_door_model_file = _this_file.parent.joinpath("../assets/config_files/cabinet_models_door.yml")
# cabinet_door_ids = get_model_ids_from_yaml(cabinet_door_model_file)
cabinet_door_infos = get_raw_yaml(cabinet_door_model_file)

for cabinet_id in cabinet_door_infos:
    register(
        id='OpenCabinetDoor_{:s}-v0'.format(cabinet_id),
        entry_point='mani_skill.env.open_cabinet_door_drawer:OpenCabinetDoorEnv',
        kwargs={'variant_config': {"partnet_mobility_id": cabinet_id},
        }
    )

    for fixed_target_link_id in range(cabinet_door_infos[cabinet_id]['num_target_links']):
        register(
            id='OpenCabinetDoor_{:s}_link_{:d}-v0'.format(cabinet_id, fixed_target_link_id),
            entry_point='mani_skill.env.open_cabinet_door_drawer:OpenCabinetDoorEnv',
            kwargs={'variant_config': {"partnet_mobility_id": cabinet_id},
                    'fixed_target_link_id': fixed_target_link_id,
            }
        )

################################################################
# OpenCabinetDrawer 
################################################################

register(
    id='OpenCabinetDrawer-v0',
    entry_point='mani_skill.env.open_cabinet_door_drawer:OpenCabinetDrawerEnv',
)

cabinet_drawer_model_file = _this_file.parent.joinpath("../assets/config_files/cabinet_models_drawer.yml")
# cabinet_drawer_ids = get_model_ids_from_yaml(cabinet_drawer_model_file)
cabinet_drawer_infos = get_raw_yaml(cabinet_drawer_model_file)

for cabinet_id in cabinet_drawer_infos:
    register(
        id='OpenCabinetDrawer_{:s}-v0'.format(cabinet_id),
        entry_point='mani_skill.env.open_cabinet_door_drawer:OpenCabinetDrawerEnv',
        kwargs={'variant_config': {"partnet_mobility_id": cabinet_id},
        }
    )

    for fixed_target_link_id in range(cabinet_drawer_infos[cabinet_id]['num_target_links']):
        register(
            id='OpenCabinetDrawer_{:s}_link_{:d}-v0'.format(cabinet_id, fixed_target_link_id),
            entry_point='mani_skill.env.open_cabinet_door_drawer:OpenCabinetDrawerEnv',
            kwargs={'variant_config': {"partnet_mobility_id": cabinet_id},
                    'fixed_target_link_id': fixed_target_link_id,
            }
        )


################################################################
# PushChair 
################################################################

register(
    id='PushChair-v0',
    entry_point='mani_skill.env.push_chair:PushChairEnv',
)

chair_model_file = _this_file.parent.joinpath(
    "../assets/config_files/chair_models.yml")
chair_ids = get_model_ids_from_yaml(chair_model_file)

for chair_id in chair_ids:
    register(
        id='PushChair_{:s}-v0'.format(chair_id),
        entry_point='mani_skill.env.push_chair:PushChairEnv',
        kwargs={'variant_config': {"partnet_mobility_id": chair_id},
        }
    )


################################################################
# MoveBucket 
################################################################

register(
    id='MoveBucket-v0',
    entry_point='mani_skill.env.move_bucket:MoveBucketEnv',
)

bucket_model_file = _this_file.parent.joinpath("../assets/config_files/bucket_models.yml")
bucket_ids = get_model_ids_from_yaml(bucket_model_file)

for bucket_id in bucket_ids:
    register(
        id='MoveBucket_{:s}-v0'.format(bucket_id),
        entry_point='mani_skill.env.move_bucket:MoveBucketEnv',
        kwargs={'variant_config': {"partnet_mobility_id": bucket_id},
        }
    )

################################################################
# Custom Split Example
################################################################

register(
    id='PushChair_CustomSplit-v0',
    entry_point='mani_skill.env.push_chair:PushChairEnv',
    kwargs={'override_model_file': ('chair', 'chair_models_custom_split_example.yml')}
)
