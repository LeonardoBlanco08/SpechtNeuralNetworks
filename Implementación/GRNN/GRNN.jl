# Módulos usados
using   Statistics
using Clustering

# Definir función de kernel gaussiano
function K_gauss(x, x0, h)
    d = length(x0)
    u = transpose(x - x0) * (x - x0) / h^2
    kernel = (2π)^(-d/2) * h^(-d) * exp(-u / 2)
    return kernel
end

# Función para centrar y estandarizar los datos
function estandarizar!(x)
    σ = std(x, dims=1)
    x .= x ./ σ
    return x, σ
end


# Definir función para fijar un tipo de kernel escalado con su factor de suavizamiento
function fija_Kh(h, tipo)
    if tipo == "gaussiano"
        Kh = (x, x0) -> K_gauss(x, x0, h) / h
    end
    return Kh
end

# Función para entrenar GRNN
function entrenaGRNN(y0, x0, h, tipo="gaussiano", agrupa=false, k=nothing)
    if agrupa
        k = Int(k)
        clustering = kmeans(x0', k)
        c0    = clustering.centers' # Centroides por clúster
        n0 = clustering.counts  # Cantidad de observaciones por clúster
        y0 = [sum(y0[clustering.assignments .== i]) for i in 1:k]  # Suma por clúster
    else
        c0 = x0
        n0 = ones(Int, size(x0, 1))  # Si no se agrupan, todas las observaciones pertenecen a clústers distintos
    end
    Kh = fija_Kh(h, tipo)
    return (y0, c0, Kh, n0)
end

# Función para predecir con GRNN
function prediceGRNN(x1, modelo)
    (y0, x0, Kh, n0) = modelo
    n = length(y0)
    m = size(x1, 1)
    y1 = Vector{Float64}(undef, m) # Inicializar el vector y1
    for j in 1:m
        suma_abajo = 0.0
        suma_arriba = 0.0
        x = x1[j, :]   
        for i in 1:n
            # Capa de patrones
            kernel = Kh(x, x0[i, :])
            # Capa de suma
            suma_abajo += n0[i] * kernel 
            suma_arriba += y0[i] * kernel
        end
        if abs(suma_abajo) > 1e-30
            y1[j] = suma_arriba / suma_abajo
        else
            y1[j] = 0.0
        end
    end
    return y1
end

# Función principal para GRNN
function GRNN(x1, y0, x0, h, tipo="gaussiano", estandariza=true, agrupa=false, k=nothing)
    if estandariza
        # Estandarizar x1 y x0
        x0, σ = estandarizar!(x0)
        x1 = x1 ./ σ
        
    end
    # Capa de entrada
    modelo = entrenaGRNN(y0, x0, h, tipo, agrupa, k)
    # Capas de patrones y sumas
    y1 = prediceGRNN(x1, modelo)
    # Capa de salida
    return y1
end
