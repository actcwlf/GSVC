@dataclass
class EncodeMeta:
    total_anchor_num: int
    anchor_num: int
    batch_size: int
    anchor_infos_list: typing.List[typing.Any]
    min_feat_list: typing.List[typing.Any]
    max_feat_list: typing.List[typing.Any]
    min_scaling_list: typing.List[typing.Any]
    max_scaling_list: typing.List[typing.Any]
    min_offsets_list: typing.List[typing.Any]
    max_offsets_list: typing.List[typing.Any]