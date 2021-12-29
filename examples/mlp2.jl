using PyCall
py"""import validate_candidate"""
iter = [t.iteration for t in trace]
data = [ trace[i+1,1,1].metadata["pop"] for i in iter ]
evo_loss
model = Chain(Dense(d, 15, relu), Dense(15, nclasses))
@info "MLP" loss=loss(data, evomodel) accuracy = accuracy(data, evomodel)

function loss(model)
    σee = model[1]
    pee = model[2]
    σei = model[3]
    pei = model[4]
    P1, C1 = make_net(Ne, Ni, σee = σee, pee = pee, σei = σei, pei = pei)#,a=a)
    E1, I1 = P1
    SNN.monitor([E1, I1], [:fire])
    sim_length = 500
    @inbounds for t = 1:sim_length*ms
        E1.I = vec([11.5 for i = 1:sim_length])#vec(E_stim[t,:])#[i]#3randn(Ne)
        SNN.sim!(P1, C1, 1ms)
    end

    spkd_found = get_trains(P1[1])
    println("Ground Truth \n")
    SNN.raster([E]) |> display
    println("Best Candidate \n")

    SNN.raster([E1]) |> display

    error = raster_difference(spkd_ground, spkd_found)
    error
end
#loss(model) = (x,y)->logitcrossentropy(model(x), y)
#loss(model,x,y) = loss(model)(x, y)
#loss(xy, model) = loss(model)(hcat(map(first,xy)...), hcat(map(last,xy)...))


opt = ADAM(1e-4)
evalcb = Flux.throttle(() -> @show(loss(data, model), accuracy(data, model)), 5)
for i in 1:500
    Flux.train!(loss(model), params(model), data, opt, cb = evalcb)
end
f_nn = (x1,x2) -> Tracker.data(mod([x1;x2]))[1]
#
plot(x1,x2,f_nn,seriestype=:surface,color=:blues)
plot!(X[1,:],X[2,:],Y',seriestype=:scatter,markersize=MS1,marker=M1,markercolor=MC1,label="data")
plot!(xlabel="Weight [kg]", ylabel="Age [years]", zlabel="Blood fat content")
plot!(title="Humans: blood fat vs. weight, age",camera=(-60,20))
