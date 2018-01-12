{lib, fetchFromGitHub, buildPythonPackage, numpy, protobuf3_3, six, pytest, }:

buildPythonPackage rec {
  pname = "tensorboard-pytorch";
  rev = "v0.9";
  name = "${pname}-${rev}";

  src = fetchFromGitHub {
    owner = "lanpa";
    repo = "tensorboard-pytorch";
    inherit rev;
    sha256 = "1hkjalap81bg17jr322kac440ci36ywpg1z0k9dzp90yas3gffjs";
  };

  buildInputs = [ pytest ];
    
  propagatedBuildInputs = [ numpy protobuf3_3 six ];

  doCheck = false;
  
  meta = {
    description = "tensorboard for pytorch";
    homepage = https://github.com/lanpa/tensorboard-pytorch;
    maintainers = with lib.maintainers; [ lebastr ];
  };
}
