{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pandas
    pkgs.python311Packages.scikit-learn
    pkgs.python311Packages.requests
    pkgs.python311Packages.pyyaml
    pkgs.python311Packages.pydantic
    pkgs.sqlite
  ];
}
