from dataloader import get_loader
from configs import get_config, load_json
import os
import solvers


def main():
    config = get_config(mode='train')
    val_config = get_config(mode='valid')
    with open(os.path.join(config.save_path, 'config.json'), 'w') as json_f:
        config.to_json(json_f)

    raw_data = load_json(config.all_path)
    train_data_loader = get_loader(raw_data=raw_data, max_len=config.max_len, batch_size=config.batch_size,
                                   shuffle=True, user_map_dict=config.user_map_dict, max_users=config.max_users)

    raw_data = load_json(val_config.all_path)
    eval_data_loader = get_loader(raw_data=raw_data, max_len=val_config.max_len, batch_size=val_config.eval_batch_size,
                                  shuffle=False, user_map_dict=config.user_map_dict, max_users=config.max_users)

    model_solver = getattr(solvers, "Solver{}".format(config.model))
    solver = model_solver(config, train_data_loader, eval_data_loader, is_train=True)

    solver.build()
    solver.train()
    solver.writer.close()

    return config


if __name__ == '__main__':
    main()
