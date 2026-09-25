# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.experiment import Experiment
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType
from benchpark.directives import variant, maintainers

class Ffb(
    Experiment,
    ProgrammingModel(
        ProgrammingModelType.Mpionly,
    ),
):
    variant(
        "workload",
        default="cavity",
        description="ffb",
    )

    variant(
        "version",
        default="67.01",
        description="Which benchmark version to use.",
    )

    maintainers("ando")

    def compute_applications_section(self):
        has_cuda = self.system_spec.satisfies("compiler=cuda")

        if has_cuda: # GPU
            self.add_experiment_variable("n_nodes", 4, True)
            self.add_experiment_variable("processes_per_node", 1)
            self.add_experiment_variable("n_ranks", "{processes_per_node} * {n_nodes}")
            self.add_experiment_variable("size", 31255875, True)
            self.add_experiment_variable("extra_batch_opts", "-N 4", named=False)
        else: # CPU
            self.add_experiment_variable("n_nodes", ["4"], True)
            self.add_experiment_variable("processes_per_node", ["4"])
            self.add_experiment_variable("n_ranks", "{processes_per_node} * {n_nodes}")
            self.add_experiment_variable("omp_num_threads", ["12"])
            self.add_experiment_variable("size", 8493380, True)
            self.add_experiment_variable("extra_batch_opts", "-N 4", named=False)

        self.set_required_variables(
            n_resources="{n_ranks}",
            process_problem_size="{size}/{n_ranks}",
            total_problem_size="{size}",
        )

    def compute_package_section(self):
        base_version = self.spec.variants['version'][0]
        ret = self.system_spec.variants['cluster']
        if ret:
            cluster = f"-{ret[0]}"
        else:
            cluster = ""

        # Fugaku's build is the one with no machine suffix. The package
        # declares 67.01-cpu - whose url is ffb-frt_cpu.fugaku.tar.gz, on
        # Fugaku's own filesystem - alongside 67.01-cpu-genoa and
        # 67.01-gpu-gh200. Suffixing it the way every other riken machine
        # is suffixed asks for a version nobody declared.
        if cluster == "-fugaku":
            cluster = ""

        suffix = "-gpu" if self.system_spec.satisfies("compiler=cuda") else "-cpu"
        # `@=` and not `@`: these version names nest, and a bare `@` is a
        # range. `ffb@67.01-cpu` therefore also matches `67.01-cpu-genoa`,
        # and spack prefers that one - so a Fugaku run went looking for
        # genoa's archive on a filesystem Fugaku does not have:
        #
        #     Error: FetchError: All fetchers failed for
        #       spack-stage-ffb-67.01-cpu-genoa-...
        #     file:///.../ffb-frt_cpu.genoa.ftz.tar.gz:
        #       No such file or directory
        #
        # The same ambiguity is what turns a version that does not exist
        # into spack's otherwise baffling "Cannot satisfy
        # 'ffb@67.01-cpu-fugaku' 1(67.01-gpu-gh200)". `@=` pins the exact
        # version, which is what naming a machine's archive meant all along.
        spec_str = f"ffb@={base_version}{suffix}{cluster}"
        self.add_package_spec(self.name, [spec_str])

