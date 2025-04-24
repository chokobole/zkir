load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

HEADERS = [
    "omp.h",
    "ompx.h",
    "omp-tools.h",
    "ompt.h",
]

# NOTE: This is only for macos.
#       On other platforms openmp should work without local config.
genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = """
      mkdir -p $(@D)/
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /opt/homebrew/opt/libomp/include/$$file $(@D)/$$file
      done
    """,
)

cc_library(
    name = "omp",
    hdrs = HEADERS,
    include_prefix = "third_party/omp/include",
    includes = ["."],
    linkopts = [
        "-L/opt/homebrew/opt/libomp/lib",
        "-lomp",
    ],
)
