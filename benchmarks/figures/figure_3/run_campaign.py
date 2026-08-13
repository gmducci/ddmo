"""Run or resume the strict publication Figure 3 campaign."""

from pathlib import Path

from benchmarks.common.run_campaign import campaign_parser, run_campaign


def main() -> None:
    parser = campaign_parser(__doc__)
    parser.set_defaults(workers=5)
    args = parser.parse_args()
    config_path = (
        args.config.resolve()
        if args.config
        else Path(__file__).with_name("config.toml").resolve()
    )
    run_campaign(
        config_path=config_path,
        runner_module="benchmarks.figures.figure_3.run",
        postprocess_module="benchmarks.figures.figure_3.postprocess",
        workers=args.workers,
        label="Figure 3",
    )


if __name__ == "__main__":
    main()
