<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://github.com/evelynmitchell/DimensionMixer/raw/master/images/DM.png"
      >
    </a>
  </p>
</div>

## 👋 Hello

Dimension Mixer model.


## 💻 Install

You can install `DimensionMixer` with pip in a
[**Python>=3.10**](https://www.python.org/) environment.

!!! example "pip install (recommended)"

    === "headless"
        The headless installation of `DimensionMixer` is designed for environments where graphical user interfaces (GUI) are not needed, making it more lightweight and suitable for server-side applications.

        ```bash
        pip install git+https://github.com/evelynmitchell/DimensionMixer.git
        ```
        


!!! example "git clone (for development)"
<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://github.com/evelynmitchell/DimensionMixer/raw/master/images/DM.png"
      >
    </a>
  </p>
</div>


!!! example "git clone (for development)"

    === "virtualenv"

        ```bash
        # clone repository and navigate to root directory
        git clone https://github.com/evelynmitchell/DimensionMixer.git
        cd DimensionMixer

        # setup python environment and activate it
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip

        # headless install
        pip install -e "."

        # desktop install
        pip install -e ".[desktop]"
        ```

    === "poetry"

        ```bash
        # clone repository and navigate to root directory
        git clone https://github.com/evelynmitchell/DimensionMixer.git
        cd DimensionMixer

        # setup python environment and activate it
        poetry env use python3.10
        poetry shell

        # headless install
        poetry install

        # desktop install
        poetry install --extras "desktop"
        ```
