# Notes for Developers

This section provides details on how to configure and/or develop this codebase.

## Continuos Integration/Continuous Deployment (CI/CD) Workflow

This project uses *GitHub Workflows* to automate or ease a number of development tasks.  These
workflows can be found within the `./.github/workflows/` directory and include:
1. **pull_request.yml**

    This workflow is run whenever a pull request is opened, updated or reopened.  The following things are enforced by this workflow:

    - linting and code formatting standards
    - proper maintainance of the Poetry project
    - successful building of the project
    - successful building of the documentation
    - successful running of tests

2. **bump.yml**

    This workflow leverages the colocated `bump.sh` bash script to automatically increment the project version whenever code is pushed to the `main` branch.  It is controlled by adding the text `[version:minor]` or `[version:major]` to the message of the pull request's head commit.

3. **publish.yml**

    This workflow is run whenever a new release is generated through *GitHub* (see below for details on how to do this).  Documentation is updated on *GitHub Pages* and a new version of the code is published on the *Python Package Index* (*PyPI*).

## Setting-up the Code

A local development copy of the code base can be obtained and configured as follows:

* Navigate to the *GitHub* page hosting the project
* If you want to fork the code so that you work on your own version of the repository (not generally needed or
  recommended):
    - Click on the `fork` button at the top of the page;
    - Edit the details you want to have for the new repoitory; and
    - Press `Create fork`.
* Obtain the URL for the repository you're going to use (denoted `<url>`) by clicking on the green `Code` button on the repository *GitHub* page
* On your local machine, navigate with your terminal to the location where you want to place the code
* Generate a local copy using `git clone <url>`;

!!! note
    Although not strictly necessary, it is recommended that you configure the branch permissions of any forked repositories as detailed in the *GitHub* configuration section below.

## Poetry and Python environments for development

Poetry is used to manage this project ([see here for an introduction](https://python-poetry.org)).  It simplifies & helps with managing the following:

1. **Creation and activation of a Python environment for the project**

    Python development should always be managed using a Python environment.  Poetry makes this easy for you.  You simply run the following from within the project:

    ``` console
    $ poetry shell
    ```
    
    !!! note
        You don't have to use Poetry to manage your Python environment if you would rather not.  You can instruct Poetry to respect your Python environemnts (e.g. created with `pyenv`) by setting the following option:
        ``` console
        $ poetry config virtualenvs.prefer-active-python true
        ```

2. **Dependency management**

    Poetry manages a "lock file" (which should be committed and maintained within the code repository) ensuring repeatible installs for all versions.

3. **Publication of the project to the Python Package Index (*PyPI*) so that people can easily install it for themselves**

    Once properly configured, publishing to *PyPI* with Poetry is extremely easy.  This is generally managed by the CI/CD workflow for the project though, and developers should never have to manually do this.

## Installing Development Dependencies

Once the code is locally installed, development dependencies should be installed by moving to the project's root directory and executing the following:

``` console
$ poetry install
```

In what follows, it will be assumed that this has been done.

## Guidelines

In the following, we lay-out some important guidelines for developing on this codebase.

### Branches

***Development should never be conducted on the `main` branch***.  If *GitHub* has been properly configured (see below), then merges to this branch are limited to Pull Requests (PRs) only.  Once a PR is opened for the `main` branch, the project tests are run.  When it is closed and code is committed to the main branch, the project version is automatically incremented (see below).

### Versioning

Semantic versioning (i.e. a scheme that follows a `vMAJOR.MINOR.PATCH` format; see <https://semver.org> for details) is used for this project.  ***The single point of truth for the current production version is the last git tag on the main branch with a `v[0-9]*` format***.  When developing locally, the reported version will appear as `v0.0.0-dev`.

Changes are handled by a *GitHub Workflow* which increments the version and creates a new tag whenever a push occurs to the `main` branch.  This ensures that every commit on the `main` branch is assigned a unique version.  The logic by which it modifies the version is as follows:

1. if the message of the PR's head commit contains the text `[version:major]`, then `MAJOR` is incremented;
2. else if it contains the text `[version:minor]`, then `MINOR` is incremented;
3. else `PATCH` is incremented.

A `MAJOR` version change should be indicated if the PR introduces a breaking change.  A `MINOR` version change should be indicated if the PR introduces new functionality.

!!! note
    Make sure you think carefully about what type of changes you are committing.  If you are adding functionality, make sure you bump the **MINOR** version; if you are making breaking changes, make sure you bump the **MAJOR** version.  ***Users will be very thankful that you did!***

### Tests

*PyTest* is used to run tests for this codebase.  Make sure you run them before submitting any code to a PR by
executing the following from the project root directory:

``` console
$ pytest
```

Some further comments about how testing has been configured for this project:

#### Coverage

*PyTest* has been configured for this project to create a coverage report after running.  This report will inform the developer of what fraction of the code base is exercised by the tests and give a list of lines of code in each Python filename which has not been exercised by the tests run.

While not strictly enforced, we encourage developers to make sure that anything they do to the codebase does not reduce this metric.  This report can be used to inform what parts of the codebase need further testing.

### Type Hints

Type hints are used in this codebase but presently not configured to be enforced.  Developers are encouraged to use them and use *mypy* (which has been added to the list of developer dependencies to this project) to check for a host of errors that this tool can efficiently identify.  This can be done by running `mypy`, passing it the path to the code you want to check as follows:

``` console
$ mypy  --explicit-package-bases <path>
```
This can be a specific file or a path underwhich all code is checked.  To run *mypy* on the whole project codebase, simply run the following from the code's root directory:

``` console
$ mypy  --explicit-package-bases python
```

### Git Hooks

This project has been set-up with pre-configured git hooks. They should be used as a means for developers to quickly check that (at least some) of the code standards of the project are being met by commited code.  Ultimately, all standards are actually enforced by the continuous integration pipeline (see below).  Running quick checks (like linting) at the point of commiting code can save time that might otherwise be lost later (for example) at the PR or release stage when testing needs to be rigorous and policy enforcement generally fails slower.  Developers can choose to either:

1. use the git hooks defined by this project (recommended, for the reasons given above; see below for instructions),
2. not to use them, and rely purely on the CI workflow to enforce all project policies, or
3. configure their IDE of choice to manage things, in which case it is up to them to make sure that this aligns with the policies being enforced by the CI.

If developers would like to utilise the git hooks provided by this project they just need to run the following command from within the project:
``` console
$ pre-commit install
```

Some of these hooks require internet access to work.  If you are trying to commit to the
repository locally and are being prevented from doing so because you are working online, the
hooks can be ignored by using the `--no-verify` flag when running `git commit`, like so:
``` console
$ git commit --no-verify
```

Alternatively, you can disable them by running:
``` console
$ pre-commit uninstall
```

They can subsequently be re-enabled by reinstalling them.

#### Maintaining Git Hooks

The git hooks are defined in the `.pre-commit-config.yaml` file.  Specific revisions for many of the tools listed should be managed with Poetry, with syncing managed with the [sync_with_poetry](https://github.com/floatingpurr/sync_with_poetry) hook.  Developers should take care not to use git hooks to *enforce* any project policies.  That should all be done within the continuous integration workflows.  Instead: these should just be quality-of-life checks that fix minor issues or prevent the propagation of quick-and-easy-to-detect problems which would otherwise be caught by the CI later with considerably more latency.  Furthermore, ensure that the checks performed here are consistant between the hooks and the CI.  For example: make sure that any linting/code quality checks are executed with the same tools and options.

### Releases

Releases are generated through the *GitHub* UI.  A *GitHub Workflow* has been configured to do the following when a new release is produced:

1. Run the tests for the project,
2. Ensure that the project builds,
3. Rebuild the documentation on *GitHub Pages*, and
4. Publish a new version of the code on [*PyPI*](https://pypi.org/).

!!! note
    If a release is flagged as a "pre-release" through the *GitHub* interface, then documentation will not be built and the project will be published on *test.PyPI.org* instead.

#### Generating a new release

To generate a new release, do the following:

1. Navigate to the project's *GitHub* page,
2. Click on `Releases` in the sidebar,
3. Click on `Create a new release` (if this is the first release you have generated) or `Draft release` if this is a subsequent release,
4. Click on `Choose a tag` and select the most recent version listed,
5. Write some text describing the nature of the release to prospective users, and
6. Click `Publish Release`.

### Documentation

Documentation for this project is generated using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and is hosted on *GitHub Pages* for the latest release version. The documentation system is configured with the following features:

1. **Pure Markdown content with Material Design theme**

    All documentation is written in standard Markdown format, making it easy for developers to contribute and maintain.

2. **Automatic API documentation generation**

    API documentation is automatically generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/) which extracts documentation directly from the Python codebase.

3. **Mathematical notation support**

    LaTeX equations are supported using [KaTeX](https://katex.org/) for fast client-side rendering of mathematical expressions.

4. **Enhanced features**

    - Responsive Material Design theme with dark/light mode support
    - Enhanced search functionality with suggestions
    - Syntax highlighting with copy-to-clipboard functionality
    - Tabbed content and admonitions for rich formatting

#### Generating the Documentation

Documentation can be generated locally by running the following from the root directory of the project:

``` console
$ make docs
```

This will generate an HTML version of the documentation in the `site/` directory which can be opened in your browser. On a Mac (for example), this can be done by running the following:

``` console
$ open site/index.html
```

For development with live reload, use:

``` console
$ make serve
```

This will start a local development server at `http://127.0.0.1:8000` with automatic reloading when files change.

#### Editing the Documentation

The majority of documentation changes can be managed in one of the following ways:

1. **Edits to existing Markdown files**:

    Edit the Markdown files directly in the `docs/` directory. The main content files include:
    - `index.md` - Landing page
    - `getting_started.md` - Installation and setup
    - `contributing.md` - Contributing guidelines
    - `state_space.md` - Technical documentation
    - `notes_for_developers.md` - Developer notes

2. **Project Docstrings**:

    Documentation for API functions, classes, and methods should be managed directly in the docstrings of the project's `.py` files. This content will automatically be harvested by `mkdocstrings` and appear in the API reference section.

3. **Add new documentation**:

    Create new `.md` files in the `docs/` directory and add them to the navigation structure in `mkdocs.yml`. The navigation order determines how they appear in the final documentation.

4. **Examples and tutorials**:

    Add example documentation to the `docs/examples/` directory. Use the existing placeholder files as templates.

#### Adding images and assets

Place any images, plots, etc. used in the documentation in the `docs/assets/` directory. Reference them in Markdown using relative paths like `![Description](assets/image.png)`.

## Configuring Services

Develpers and project owners/maintainers will require accounts with one or all of the following services to work with this codebase.  This section details how these services need to be configured.  Following these steps should only be necessarry - or partially necessary - if a developer chooses to fork the project.

1. [***GitHub***](https:/github.com)

    To work with this codebase, you will require a *GitHub* account ([go here to get one](https://github.com)).
    
    Branch permissions for the main project repository should be configured only permit merges from pull requests.  To do so, navigate to `Settings->Branches->Add branch ruleset` and:

        - give the Ruleset whatever name you'd like (e.g. `Protect Main`)
        - set `Enforecement status` to `Active`
        - add a `Target Branch` targeting criteria by pattern and type `main`
        - select `Require a pull request before merging`
        - select `Require status checks to pass` and add `Run all build and unit tests` from GitHub Actions as a required check

    This will ensure that all CI/CD tests pass before a merge to the main branch can be made.
    
    Several secrets need to be configured by navigating to `Settings->Secrets->Actions` and adding the following:
    
    - To make code releases available on the **Python Package Index** (see below), then the following secret needs to be set (see below for where to find this value):
    
        - **PYPI_TOKEN**,
    
    - To test code releases with the **Test Python Package Index** (see below), then the following secret needs to be set (see below for where to find this value):
    
        - **TEST_PYPI_TOKEN**,

2. [__GitHub Pages__](https://pages.github.com)

    **GitHub Pages** is used to host the project documentation. No additional account setup is required beyond your existing GitHub account. The documentation is automatically deployed via GitHub Actions when code is pushed to the `main` branch.

    To enable GitHub Pages for your repository:
    
    1. Go to your repository's **Settings** → **Pages**
    2. Under **Source**, select "Deploy from a branch"
    3. Select the `gh-pages` branch (created automatically by the deployment workflow)
    4. Select `/ (root)` as the folder
    5. Click **Save**

    Once configured, documentation will be automatically updated and available at `https://username.github.io/repository-name/` whenever changes are merged to the main branch.

3. The [__Python Package Index (*PyPI*)__](https://pypi.org)

    This service is used to publish project releases.  An account is needed if you are the owner of the project, but not generally needed if you are simply a contributing developer.  An API token will need to be created and added to your *GitHub* project as **PYPI_TOKEN** (as detailed above).  This can be generated from the *PyPI* UI by navigating to `Account Settings->Add API Token`.

    To test releases, a parallel account on *test.PyPI* is needed and a similar token to **PYPI_TOKEN** - named **TEST_PYPI_TOKEN** needs to be set, in the same way as above.  To create a test release, flag it as a "pre-release" through the *GitHub* interface when you generate a release, and it will be published on *test.PyPI.org* rather than *PyPI.org*.

    !!! note
        Although `poetry` can be used to directly publish this project to *PyPI*, users should not do this.  The proper way to publish the project is through the *GitHub* interface, which leverages the *GitHub Workflows* of this project to ensure the enforcement of project standards before a new version can be created.
