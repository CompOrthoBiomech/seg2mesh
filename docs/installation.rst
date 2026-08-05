Installation
============

.. note::
    Whenever you are updating `seg2mesh`, remember to first do a `git pull` (or Download the ZIP and extract) to get the latest version.

Prerequisites
-------------

Install the `uv` package manager:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`uv` is an ultra-fast Python package manager that makes environment and dependency management much easier.

To install on Windows, in a Powershell terminal execute:

.. code-block:: powershell

    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

To install on macOS or Linux execute:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

Install git:
~~~~~~~~~~~~

**On Windows:**

You have options, but if you're new to Git, a simple solution is to install the `GitHub Desktop <https://desktop.github.com/>`_ application.

**On macOS:**

Again, you have options, but using `brew` is quite nice.

If you need to install `brew`, you can do so with,

.. code-block:: bash

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

You can then install `git` with,

.. code-block:: bash

    brew install git

**On Linux:**

Using your distribution's package manager is the most straightforward option.

For example, on Debian/Ubuntu flavors, you can use `apt` with,

.. code-block:: bash

    sudo apt install git

On Arch-based flavors, you can use `pacman` with,

.. code-block:: bash

    sudo pacman -S git


Clone the seg2mesh repository:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have an ssh key set up for GitHub, clone the repository using ssh:

.. code-block:: bash

    git clone git@github.com:seg2mesh/seg2mesh.git

If you don't have an ssh key set up, clone the repository using https:

.. code-block:: bash

    git clone https://github.com/seg2mesh/seg2mesh.git

Optional -- If you really don't want Git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can download the repository as a ZIP file, by clicking on the green rectangle with the "< >"
and selecting "Download ZIP".

Install or Update seg2mesh:
---------------------------

After completing the prerequisites, you can install seg2mesh by running the following command:

.. code-block:: bash

    uv sync

From anywhere in the repository root directory or below.

If you cloned the repository using Git, you can update to the latest version with:

.. code-block:: bash

    git pull

and then run,

.. code-block:: bash

    uv sync

**Congratulations! You have successfully installed seg2mesh.**
