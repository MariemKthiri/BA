module SCM3GPP

using Distributions
using Utils # for crandn

include("toeplitz_helpers.jl")
include("scm_helpers.jl")
include("SCM3GPPMacro.jl")
include("SCMMulti.jl")

end
