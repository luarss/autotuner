import uuid
from enum import Enum
from typing import Self

from dotenv import load_dotenv
from pydantic import BaseModel, DirectoryPath, Field, FilePath, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class TuneEvalFunction(Enum):
    DEFAULT = "default"
    PPA_IMPROV = "ppa-improv"


class TuneConfig(BaseModel):
    algorithm: str = ""
    eval: TuneEvalFunction = Field(default=TuneEvalFunction.DEFAULT)
    samples: int = 5
    iterations: int = 1
    resources_per_trial: int = 1
    reference: FilePath | None = None
    reference_dict: dict = {}
    perturbation: int = 25
    seed: int | None = None
    config: FilePath | None = None
    config_dict: dict = {}
    resume: bool | None = None

    @model_validator(mode="after")
    def check_ppa_improve_requires_reference(self) -> Self:
        if self.eval == "ppa-improv" and not self.reference:
            raise ValueError("[ERROR TUN-0006] PPAImprov requires a reference file.")
        return self


class SweepConfig(BaseModel):
    config: FilePath | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Environment Config
    metric: str = Field(default="metric")
    error_metric: float = Field(default=9e99)
    orfs_flow_dir: DirectoryPath | None = None
    local_dir: DirectoryPath | None = None
    install_path: DirectoryPath | None = None
    sdc_original: FilePath | None = None
    fr_original: FilePath | None = None

    # Design Config
    design: str = "gcd"
    platform: str = "asap7"
    experiment: str = Field(default="test")
    timeout: float | None = Field(default=3600)
    verbose: int = 0
    seed: int = Field(default=42)

    # Workload Config
    jobs: int = 1
    openroad_threads: int = 16
    server: str | None = None
    port: int = 10001

    # Mode Config
    mode: str
    tune: TuneConfig = Field(default_factory=lambda: TuneConfig())
    sweep: SweepConfig = Field(default_factory=lambda: SweepConfig())

    @model_validator(mode="after")
    def check_resume_requires_experiment(self) -> Self:
        if self.mode == "tune":
            if self.tune.resume and not self.experiment == "":
                raise ValueError("[ERROR TUN-0031] Resume requires a non-default experiment name to be set.")
        return self

    @model_validator(mode="after")
    def process_experiment_name(self) -> Self:
        """
        Processes the experiment name after all other fields are validated.
        - If the name is "default", creates a new name like "mode-uuid".
        - If a custom name is provided, appends "-mode" to it.
        """
        default_experiment_name = self.model_fields["experiment"].default

        if self.experiment == default_experiment_name:
            unique_id = str(uuid.uuid4())[:8]
            self.experiment = f"{self.mode}-{unique_id}"
        else:
            self.experiment = f"{self.experiment}-{self.mode}"

        return self
