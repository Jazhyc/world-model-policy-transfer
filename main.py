from dreamerv3.train import make_env, make_envs, make_replay, make_logger
from dreamerv3 import embodied
from functools import partial as bind


def main(argv=None):
    from . import agent as agt

    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(agt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    args = embodied.Config(
            **config.run, logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length)
    print(config)

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)

    cleanup = []
    try:

        if args.script == 'train':
            replay = make_replay(config, logdir / 'replay')
            env = make_envs(config)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train(agent, env, replay, logger, args)

        elif args.script == 'train_save':
            replay = make_replay(config, logdir / 'replay')
            env = make_envs(config)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_save(agent, env, replay, logger, args)

        elif args.script == 'train_eval':
            replay = make_replay(config, logdir / 'replay')
            eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
            env = make_envs(config)
            eval_env = make_envs(config)    # mode='eval'
            cleanup += [env, eval_env]
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_eval(
                    agent, env, eval_env, replay, eval_replay, logger, args)

        elif args.script == 'train_holdout':
            replay = make_replay(config, logdir / 'replay')
            if config.eval_dir:
                assert not config.train.eval_fill
                eval_replay = make_replay(config, config.eval_dir, is_eval=True)
            else:
                assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
                eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
            env = make_envs(config)
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_holdout(
                    agent, env, replay, eval_replay, logger, args)

        elif args.script == 'eval_only':
            env = make_envs(config)    # mode='eval'
            cleanup.append(env)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            embodied.run.eval_only(agent, env, logger, args)

        elif args.script == 'parallel':
            assert config.run.actor_batch <= config.envs.amount, (
                    config.run.actor_batch, config.envs.amount)
            step = embodied.Counter()
            env = make_env(config)
            agent = agt.Agent(env.obs_space, env.act_space, step, config)
            env.close()
            replay = make_replay(config, logdir / 'replay', rate_limit=True)
            embodied.run.parallel(
                    agent, replay, logger, bind(make_env, config),
                    num_envs=config.envs.amount, args=args)

        else:
            raise NotImplementedError(args.script)
    finally:
        for obj in cleanup:
            obj.close()


if __name__ == "__main__":
    main()