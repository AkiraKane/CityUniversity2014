Pkg.add("MetadataTools")
Pkg.test("MetadataTools")
using MetadataTools
pkgs = get_all_pkg()