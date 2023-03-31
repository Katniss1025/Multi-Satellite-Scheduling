import yaml

def read_config_file(config_parser):
    args = config_parser.parse_args()
    config_file_path = 'config_files/' + args.config_file
    with open(config_file_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        config_parser.set_defaults(**cfg)
    args = config_parser.parse_args()
    return args

