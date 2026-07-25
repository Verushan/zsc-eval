import argparse
import os
import os.path as osp

from loguru import logger

policy_pool_dir = os.getenv("POLICY_POOL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=str)
    parser.add_argument("alg", type=str)
    parser.add_argument("sp1_model_path", type=str)
    parser.add_argument("sp2_model_path", type=str)

    args = parser.parse_args()

    if args.layout == "all":
        layouts = [
            "random0",
            "random0_medium",
            "random1",
            "random3",
            "small_corridor",
            "unident_s",
            "random0_m",
            "random1_m",
            "random3_m",
        ]
    else:
        layouts = [args.layout]

    for layout in layouts:
        yml_dir = osp.join(policy_pool_dir, layout, args.alg)
        logger.info(f"Using yaml directory of: {yml_dir}")

        os.makedirs(yml_dir, exist_ok=True)

        sp1_model_base_path = os.path.basename(os.path.normpath(args.sp1_model_path))
        sp2_model_base_path = os.path.basename(os.path.normpath(args.sp2_model_path))

        yml_path = osp.join(
            policy_pool_dir,
            layout,
            args.alg,
            f"{sp1_model_base_path}&{sp2_model_base_path}.yml",
        )

        logger.info(f"Using yaml path of: {yml_path}")

        yml = open(
            yml_path,
            "w",
            encoding="utf-8",
        )

        yml.write(
            f"""s1:
            policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
            featurize_type: ppo
            train: False
            model_path:
                actor: {args.sp1_model_path}
        """
        )

        yml.write(
            f"""s2:
            policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
            featurize_type: ppo
            train: False
            model_path:
                actor: {args.sp2_model_path}
        """
        )

        yml.close()


if __name__ == "__main__":
    main()
