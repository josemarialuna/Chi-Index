# Releasing

Releases are explicit maintainer actions. The normal CI workflow never publishes.
Publishing a non-prerelease GitHub release runs `publish.yml`: it checks the tag
against package metadata, runs tests, builds both distributions, validates them,
then uploads through PyPI Trusted Publishing. No long-lived API token is stored
in the repository.

Configure a trusted publisher once in the existing `chi-index` project's PyPI
publishing settings with:

- Owner: `josemarialuna`
- Repository: `Chi-Index`
- Workflow filename: `publish.yml`
- GitHub environment: `pypi`

Only a PyPI project owner can authorize this publisher. GitHub credentials do
not grant PyPI ownership. The publish job has OIDC permission; the build job
does not. The publisher must exactly match the workflow and environment names.

Before a release, update the version in `pyproject.toml` and date the changelog.
Merge the reviewed PR after CI passes. Create a tag `v<VERSION>` on that commit,
then publish the GitHub release with migration notes. Confirm the publish
workflow passes and install the new version from PyPI in a clean environment.

If authorization is missing, configure the publisher and rerun the failed
workflow. Inspect PyPI first if an upload was interrupted: published files and
version numbers cannot be replaced. Never reuse a version for changed code.

Reference: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
