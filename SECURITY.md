# Cqlib Security Policy

[简体中文](SECURITY.CN.md)

Cqlib takes project and user security seriously. If you discover a security vulnerability, supply-chain risk, or issue that may affect user runtime environments, do not open a public issue, pull request, or discussion.

Please report it privately by emailing <tianyan@chinatelecom.cn>. The project website is <https://qc.zdxlz.com/>.

## Reporting Scope

Please use the security email for issues such as:

- Potential arbitrary code execution, privilege escalation, or data disclosure
- Supply-chain risks in dependencies, build, release, or installation workflows
- Memory-safety issues in Python bindings, C bindings, or FFI boundaries
- Denial-of-service risks when parsing external input, loading files, or processing quantum circuits or IR
- Issues that may affect package integrity, build artifact trust, or user environment security

For ordinary bugs, feature requests, documentation issues, or non-security compatibility problems, use the normal issue or pull request workflow.

## Report Contents

To help us confirm and fix the issue, include as much of the following as possible:

- Affected version, commit, or release package
- Affected platform, operating system, CPU architecture, Rust version, and Python version
- Vulnerability type and impact
- Minimal reproduction steps or runnable example
- Relevant logs, crash information, stack trace, or input sample
- Any mitigation or fix you suggest
- Whether the issue has already been reported to other platforms, organizations, or databases

Do not include unrelated personal data, production secrets, real user data, or third-party materials you are not authorized to share.

## Handling Process

Maintainers will acknowledge reports as soon as practical, then assess, fix, and release according to the impact of the issue. During the process, maintainers may contact the reporter for reproduction details, fix verification, or coordinated disclosure timing.

Please do not publicly disclose vulnerability details, exploit code, or reproducible attack steps before a fix is released, so users are not put at unnecessary risk.

## Supported Versions

The project is still in an early version stage. In general, security fixes will be prioritized for the current main branch and the latest release. Whether older versions receive backported fixes depends on severity, fix complexity, and maintenance cost.

## Credit

If the reporter wants public credit, maintainers may acknowledge the report after a fix is released through release notes, an acknowledgements list, or another suitable channel. Public credit depends on the reporter's preference and the disclosure plan.
