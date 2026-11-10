# SPDX-License-Identifier: GPL-3.0-or-later
# Sourced by the build scripts: run_check runs one source or package check. With
# EDEN_CHECKS_REPORT_ONLY=1 a failing check is reported (a workflow warning, and a line in
# build/check-failures.txt) and the build goes on, so one run gives the package and every
# check's result; otherwise a failing check stops the build, as before.
run_check() {
    if [[ ${EDEN_CHECKS_REPORT_ONLY:-0} != 1 ]]; then
        "$@"
        return
    fi
    if ! "$@"; then
        echo "::warning title=Check failed (report only)::$*"
        mkdir -p "$root/build"
        echo "$*" >> "$root/build/check-failures.txt"
    fi
}
