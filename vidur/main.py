from vidur.config import SimulationConfig
from vidur.simulator import Simulator


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()

    simulator = Simulator(config)
    simulator.run()


if __name__ == "__main__":
    main()
