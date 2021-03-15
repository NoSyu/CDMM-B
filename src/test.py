from dataloader import get_loader
from configs import get_config, load_json
import solvers
import nosyupylib
import traceback
import os
import re


def main():
    config = get_config(mode='test')

    with open(os.path.join(config.save_path, 'config.json'), 'r') as json_f:
        temp_config_str = json_f.read()
        config.max_users = int(re.findall(r"'max_users': ([0-9]+?),", temp_config_str)[0])
        config.max_len = int(re.findall(r"'max_len': ([0-9]+?),", temp_config_str)[0])
        config.rnn_hidden_size = int(re.findall(r"'rnn_hidden_size': ([0-9]+?),", temp_config_str)[0])

    raw_data = load_json(config.all_path)
    test_data_loader = get_loader(raw_data=raw_data, max_len=config.max_len, batch_size=config.batch_size,
                                  shuffle=False, user_map_dict=config.user_map_dict, max_users=config.max_users)

    model_solver = getattr(solvers, "Solver{}".format(config.model))
    solver = model_solver(config, None, test_data_loader, is_train=False)

    solver.build()
    solver.test()

    return config


if __name__ == '__main__':
    add_msg = ""
    main_config = "WHAT?!"
    try:
        main_config = main()
    except Exception as e:
        add_msg = traceback.format_exc()
        pass
    finally:
        final_msg = "{}\n{}\n{}".format(__file__, str(main_config), add_msg)
        nosyupylib.alert_end_program(final_msg)
