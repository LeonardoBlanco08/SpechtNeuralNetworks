using StatsBase
using Optim
using Plots
using DataFrames
using Statistics
using CSV
using Distributions
using RDatasets
using Random

# Funciones para el calculo de h (bandwith) ----------------------------------

function bw_scott(data::AbstractVector)
    std_dev = std(data, corrected=true)
    n = length(data)
    return 3.49 * std_dev * n ^ (-1/3)
end

function _select_sigma(x::AbstractVector)
    normalizer = 1.349
    iqr = (percentile(x, 75) - percentile(x, 25)) / normalizer
    std_dev = std(x, corrected=true)
    return iqr > 0 ? min(std_dev, iqr) : std_dev
end

function bw_silverman(data::AbstractVector)
    sigma = _select_sigma(data)
    n = length(data)
    return 0.9 * sigma * n ^ (-1/5)
end

function mlcv(h, data, x, k, n)
    fj = zeros(n)
    for j in eachindex(fj)
        for i in eachindex(data)
            if i == j continue end
            fj[j] += k((x[j] - data[i]) / h)
        end
        fj[j] /= (n - 1) * h
    end
    return -mean(log.(fj[fj .> 0]))
end

function bw_mlcv(data::AbstractVector, k::Function)
    n = length(data)
    x = range(minimum(data), stop=maximum(data), length=n)
    objective(h) = mlcv(h, data, x, k, n)
    result = optimize(objective, 0.1, 10.0)  # Provide bounds for h
    h = Optim.minimizer(result)
    if abs(h) > 10
        return bw_scott(data)
    end
    return h
end

# Funcion kernel --------------------------------------------------------------

function kernel(k::String)
    function bounded(f)
        return x -> abs(x) <= 1 ? f(x) : 0.0
    end

    if k == "gaussian"
        return x -> (2pi)^(-1/2) * exp(-1/2 * x^2)
    elseif k == "epanechnikov"
        return bounded(x -> (3/4) * (1 - x^2))
    elseif k == "cosine"
        return bounded(x -> (pi / 4) * cos(pi / 2 * x))
    elseif k == "linear"
        return bounded(x -> 1 - abs(x))
    elseif k == "uniform"
        return bounded(x -> 1/2)
    elseif k == "biweight"
        return bounded(x -> (15/16) * (1 - x^2)^2)
    elseif k == "triweight"
        return bounded(x -> (35/32) * (1 - x^2)^3)
    elseif k == "triangular"
        return bounded(x -> 1.0 - abs(x))
    elseif k == "picard"
        return x -> exp(-abs(x)) / 2.0
    elseif k == "cauchy"
        return x -> 1 / (pi * (1 + x^2))
    elseif k == "fejer_de_vallee_poussin"
        return x -> (2pi)^(-1) * (sin(x / 2) / (x / 2))^2
    elseif k == "parabolic"
        return x -> abs(x) <= sqrt(5) ? (3 / (4sqrt(5))) * (1 - (x^2 / 5)) : 0.0
    elseif k == "mexican_hat"
        return x -> (3 / (2sqrt(2pi))) * (1 - (x^2 / 3)) * exp(-1/2 * x^2)
    else
        throw(ArgumentError("Unknown kernel."))
    end
end

# Funcion KDE (para estimar densidad)
function kde(data::AbstractVector, k::Function, h::Float64, x::AbstractVector)
    n = length(data)
    kde = zeros(length(x))
    for j in eachindex(x)
        for i in eachindex(data)
            kde[j] += k((x[j] - data[i]) / h)
        end
        kde[j] /= n * h
    end
    return kde
end

# Funcion KDE multivariado (especificamente kernel producto)
function multivariate_kde(data::AbstractMatrix, x_grid::AbstractMatrix, k::Function=kernel("gaussian"), h_method::String="silverman")
    n, d = size(data)
    # Selecciona el parametro h
    if h_method == "scott"
        h = [bw_scott(data[:, j]) for j in 1:d]
    elseif h_method == "silverman"
        h = [bw_silverman(data[:, j]) for j in 1:d]
    elseif h_method == "mlcv"
        h = [bw_mlcv(data[:, j], k) for j in 1:d]
    else
        throw(ArgumentError("Unknown bandwidth selection method."))
    end
    
    # Kernel producto 
    kernels_1d = [kde(data[:, j], k, h[j], [x_grid[j]]) for j in 1:d]
    product_kernel = reduce((a, b) -> a .* b, [kde(data[:, j], k, h[j], [x_grid[j]]) for j in 1:d])

    return product_kernel[1]
end

# Funcion para normalizar las variables
function normalize(vector::AbstractMatrix; mean_val=nothing, std_val=nothing)
    mean_val = isnothing(mean_val) ? mean(vector, dims=1)[:] : mean_val
    std_val = isnothing(std_val) ? std(vector, dims=1)[:] : std_val
    norm_vector = (vector .- mean_val') ./ std_val'
    return norm_vector, mean_val, std_val
end

# Funciones PNN ----------------------------------------------------------------------------------------------------

function initialize_pnn(X::AbstractMatrix, Y::AbstractVector; kernel0::String = "gaussian", sigma::Float64 = 2.0)
    kernel_func = kernel(kernel0)
    # TODO agregar parametro para que se haga solo si se solicita
    X_normalized, mean_val, std_val = normalize(X)
    return (X_normalized, Y, kernel_func, sigma, mean_val, std_val)
end

# Prediction function
function predict_pnn(pnn, X_test::AbstractMatrix)
    X, Y, kernel_func, sigma, mean_val, std_val = pnn

    # TODO agregar parametro para que se haga solo si se solicita
    X_test, _, _ = normalize(X_test; mean_val=mean_val, std_val=std_val)
    Y_pred = Vector{Any}()

    # Itera por cada observacion de la tabla de testing
    for x in eachrow(X_test)
        class_probability_score = Float64[]
        # Itera por cada clase 
        for class_label in unique(Y)
            class_samples = X[Y .== class_label, :]
            x = reshape(x, 1, :)
            # Calcula la densidad del punto en alguna clase especifica 
            density_estimation = multivariate_kde(class_samples, x, kernel_func, "silverman")
            # Calcula el score 
            score = (1 / sum(Y .== class_label)) * density_estimation
            push!(class_probability_score, score)
        end
        # Clasifica (segun cual probabilidad es mas alta)
        winner_class = unique(Y)[argmax(class_probability_score)]
        push!(Y_pred, winner_class)
    end
    return Y_pred
end


################################################################################
# PRUEBAS PNN

# Load the Iris dataset
iris = dataset("datasets", "iris")

# Prepare the dataset
X = Matrix(iris[:, 1:4])  # Convert feature columns to a matrix
Y = convert(Vector, iris[:, 5])  # Convert the label column to a vector
unique_labels = unique(Y)
label_map = Dict(label => i for (i, label) in enumerate(unique_labels))
Y_mapped = [label_map[label] for label in Y]

# Split the dataset into training and testing sets
Random.seed!(42)  # For reproducibility
train_indices = sample(1:length(Y), Int(0.7 * length(Y)), replace=false)
test_indices = setdiff(1:length(Y), train_indices)

X_train = X[train_indices, :]
Y_train = Y_mapped[train_indices]
X_test = X[test_indices, :]
Y_test = Y_mapped[test_indices]

# Possible kernels
kernels = ["gaussian", "epanechnikov", "cosine", "linear", "uniform", "biweight", "triweight", "triangular",
           "picard", "cauchy", "fejer_de_vallee_poussin", "parabolic", "mexican_hat"]

# Evaluate each kernel
results = DataFrame(kernel=String[], accuracy=Float64[])
for k in kernels
    print(k)
    pnn = initialize_pnn(X_train, Y_train, kernel0=k, sigma=1.0)
    Y_pred = predict_pnn(pnn, X_test)
    accuracy = mean(Y_pred .== Y_test)
    push!(results, (kernel=k, accuracy=accuracy))
end

println(results)


###############################################################################
# PRUEBAS KDE

# Data generation functions
function make_data_normal()
    x = randn(100)
    dist = x -> pdf(Normal(), x)
    return x, dist
end

function make_data_binormal()
    x = [randn(100) .- 2; randn(100) .+ 2]
    dist = x -> 0.5 * pdf(Normal(-2), x) + 0.5 * pdf(Normal(2), x)
    return x, dist
end

function make_data_exp()
    x = [rand(100); 2 .+ rand(100)]
    dist = x -> 0.5 * pdf(Exponential(), x) + 0.5 * pdf(Exponential(1/2), x .- 2)
    return x, dist
end

function make_data_uniform()
    x = [rand(100); 2 .+ rand(100)]
    dist = x -> 0.5 * pdf(Uniform(0, 1), x) + 0.5 * pdf(Uniform(2, 3), x)
    return x, dist
end

# Lists of data, kernels, and bandwidth algorithms
data_list = [
    ("Normal", make_data_normal),
    ("Bimodal (Normal)", make_data_binormal)
]

kernels = [
    ("Gaussian", kernel("gaussian")),
    ("Epanechnikov", kernel("epanechnikov"))
]

bw_algorithms = [
    ("Scott", bw_scott),
    ("Silverman", bw_silverman),
    ("MLCV", bw_mlcv)
]

mses = []

# Run KDE function
function run_kde!(ax, data, kernel, bw_algorithms)
    x, dist = data[2]()
    x_plot = range(minimum(x) * 1.05, stop=maximum(x) * 1.05, length=500)
    histogram!(ax, x, norm=true, alpha=0.2, bins=20, linecolor=:black)
    plot!(ax, x_plot, dist.(x_plot), fillrange=0, fillcolor=:silver, alpha=0.5, linewidth=0)
    scatter!(ax, x, fill(-0.02, length(x)), m=:vline, linecolor=:black)

    for bw in bw_algorithms
        if bw[1] == "MLCV"
            h = bw[2](x, kernel[2]) 
        else
            h = bw[2](x)
        end
        x_kde = kde(x, kernel[2], h, x_plot) 
        mse = mean((dist.(x_plot) .- x_kde) .^ 2)
        push!(mses, Dict(
            "data" => data[1],
            "kernel" => kernel[1],
            "bw_algorithm" => bw[1],
            "h" => round(h, digits=5),
            "mse" => round(mse * 1000, digits=5)
        ))
        plot!(ax, x_plot, x_kde, linewidth=1, label="h_{$bw[1]} = $(round(h, digits=5))")
    end

    ax[:legend] = :best
    ax[:title] = "$(data[1]), $(kernel[1])"
end

# Plot setup
fig = plot(layout=(2, 2), size=(800, 600))

# Run for each combination
for i in eachindex(data_list)
    for j in eachindex(kernels)
        ax = fig[i, j]
        run_kde!(ax, data_list[i], kernels[j], bw_algorithms)
    end

    for bw in bw_algorithms
        avg_h = mean([m["h"] for m in mses if m["data"] == data_list[i][1] && m["bw_algorithm"] == bw[1]])
        avg_mse = mean([m["mse"] for m in mses if m["data"] == data_list[i][1] && m["bw_algorithm"] == bw[1]])
        push!(mses, Dict(
            "data" => data_list[i][1],
            "kernel" => "-",
            "bw_algorithm" => bw[1],
            "h" => round(avg_h, digits=5),
            "mse" => round(avg_mse, digits=5)
        ))
    end
end

# Display and save plot
display(fig)
savefig(fig, "eval.pdf")

# Save results to CSV
df = DataFrame(mses)
CSV.write("eval.csv", df)
