#!/usr/bin/env bash

set -euo pipefail

# This helper does not start Ray. It only keeps a worker-node shell or
# scheduler job alive after the node has joined a Ray cluster with
# `ray start --address=...`.
sleep "${RAY_KEEPALIVE_SECONDS:-259200}"
