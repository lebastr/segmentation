{ pkgs ? import <nixpkgs> {}
, stdenv ? pkgs.stdenv
} :
let

  self = pkgs.python36Packages;
  inherit (self) buildPythonPackage fetchPypi;

  be = stdenv.mkDerivation {
    name = "buildenv";
    buildInputs =
    ( with pkgs;
      with self;
    [
    ffmpeg
    h5py
    imageio
    jupyter
    numpy
    opencv
    pillow
    pytorchWithCuda
    scikitimage
    scikitlearn
    scipy
    shapely
    tensorflow
    tkinter
    torchvision
    tqdm
    ]);

    shellHook = with pkgs;
      let
        gitbin = "${git}/bin/git";
      in ''
      if test -f /etc/myprofile ; then
        . /etc/myprofile
      fi

      mkdir .ipython-profile 2>/dev/null || true
      cat >.ipython-profile/ipython_config.py <<EOF
      print("Enabling autoreload")
      c = get_config()
      c.InteractiveShellApp.exec_lines = []
      c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
      c.InteractiveShellApp.exec_lines.append('%autoreload 2')
      EOF

      export QT_QPA_PLATFORM_PLUGIN_PATH=`echo ${pkgs.qt5.qtbase.bin}/lib/qt-*/plugins/platforms/`
      export PYTHONPATH="`pwd`/src:$PYTHONPATH"
      alias ipython='ipython --matplotlib=qt5 --profile-dir=.ipython-profile'
      alias tc='mypy --ignore-missing-imports src/*py'
      alias tcn="git status --porcelain | grep '^ M.*py' | awk '{print \$2}' | xargs mypy --ignore-missing-imports"

      mktags() {
        echo "Building ctags.."

        find $(echo $PYTHONPATH | tr ':' ' ') ./src -name '*\.py' |
        grep -v '/tests/' |
        ctags --filter=yes -L - |
        grep -E 'def|class' |
        sort > tags

        echo Done.
      }

      echo 'mktags - build ctags'
      echo 'ipython - run ipython shell'
      echo 'tc - run the typechecker on src/*py files'
      echo 'tcn - run the typechecker on NEW /git status/ src/*py files'
    '';
  };

in
  be
