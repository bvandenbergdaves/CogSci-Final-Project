
#################################### Hopfield Network w/ Temperature ####################################




# Code Libraries ----------------------------------------------------------


# library used to manipulate and plot circles
library("plotrix")

# library used to analyze the fit of non linear multiple regression models
library(nlstools)





# Outside links -----------------------------------------------------------

# Graph of the superellipsoid function:  https://www.desmos.com/calculator/bgzix13dmt
# Graphs of of neuron update probibilities: https://www.desmos.com/calculator/mgstr6i9ut
# Graph of simple approximation of phase diagram: https://www.desmos.com/calculator/dznd8ffnij

# Initializing Network Components -----------------------------------------


# Number of neurons in the network
N <- 1

# Number of patterns stored in the network
P <- 1

# Temperature
# i.e. measure of how far the network is from deterministic neuron updates
T <- 0

# Alpha
# i.e. measure of how overloaded the network storage capacity is
alpha <- P/N

# Blank network
network <- rep(0, N)

# Blank patterns
patterns <- matrix(data = rep(0, N), nrow = N, ncol = P)

# Blank weights
weights <- matrix(data = rep(0, N^2), nrow = N, ncol = N)

# Blank activation vector
# this vector represents the total actvation being recieved by every neuron in the network
act.vector <- rep(0, N)

# Blank stay chance vector
# this vector represents the chance for every neuron that if it gets picked, it will flip
# to the opposite state
stay.chance.vector <- rep(0, N)





# Basic Helper Functions --------------------------------------------------


# Create random state vector
random.state.vector <- function() {
  return(sample(c(-1,1), N, replace = TRUE))
}

# randomize the network
randomize.network <- function(){
  network <<- random.state.vector()
  act.vector <<- as.vector(weights %*% network)
  stay.chance.vector <<- mapply(function(state, prob) {
    if (state == 1) {
      return(prob)
    } else{
      return(1 - prob)
    }
  }
  , network,
  
  sapply(act.vector, function(act) {
    return(1 / (1 + exp(-2 * act / T)))
  }))
}

# Create pattern set
randomize.patterns <- function(){
  patterns <<- replicate(P, random.state.vector())
}

# set the network to the indexth pattern
set.as.pattern <- function(index){
  network <<- patterns[,index]
  
  # update the activation vector
  act.vector <<- as.vector(weights %*% network)
  
  # update the stay.chance.vector (by using the one.chance.vector
  # as an intermediary step)
  one.chance.vector <- sapply(act.vector, act.to.prob)
  stay.chance.vector <<- mapply(function(state, prob) {
    if (state == 1) {
      return(prob)
    } else{
      return(1 - prob)
    }
  }, network, one.chance.vector)
}

# check if the network is a pattern
is.pattern <- function(){
  
  output <- FALSE
  
  # check if the network is identical to any of the patterns
  for (i in 1:P) {
    
    # if it is for any pattern, set the output to true and
    # end the for loop
    if(identical(network, patterns[,i])){
      output <- TRUE
      i <- P+1
    }
  }
  
  # return the result
  return(output)
}


# This function gives the probability of a given neuron updating to 1 due to an activation value.
# The probability is taken from a logistic curve with coefficient 2/T in the exponent.
act.to.prob <- function(act) {
  return(1 / (1 + exp(-2 * act / T)))
}

# probabilistic neuron activation to neuron state function
act.to.state <- function(act){
  return(
    sample(c(1,-1), 1, prob = c(act.to.prob(act), 1-act.to.prob(act)))
  )
}


# This graph shows the output of the act.to.prob function for a range of activations. By testing
# different T values, we can see the effect of temperature on our model.
plot(seq(-2, 2, by = 0.01), act.to.prob(seq(-2, 2, by = 0.01)), type = 'l',
     xlab = "Neuron Activation a", ylab = "P(a)", main = "Probabilistic Update Rule When T = 0.2")


# measures the hamming dist in % neurons different of the network from a given state vector. 
hamming <- function(state.vector) {
  hamming.dist <- sum(sapply((state.vector - network), function(x){return(abs(x))}))/2
  return(hamming.dist/N)
}

# gives the energy of the network in its current state
energy <- function(state.vector) {
  E <- 0
  for (i in 1:N) {
    for (j in 1:N) {
      E <- E - (weights[i,j]*state.vector[i]*state.vector[j])/2
    }
  }
  return(E)
}

# this function represents the change in the hamiltonian resulting from a state change in the indexth neuron
energy.change <- function(index, old.state, new.state) {
  return((old.state-new.state)*(weights[index,] %*% network))
}


# Training weights
simple.train <- function(){
  # initialize blank weight matrix
  weights <<- matrix(nrow = N, ncol = N)
  
  # set the weight matrix diagonals to zero
  for (d in 1:N) {
    weights[d,d] <<- 0
  }
  
  # make a whle loop that covers the upper triangle of the
  # weight matrix, moving diagonally downwards starting at
  # w_12. The if statement at the bottom ensures that when
  # the loop reaches the bottom of the diagonal it is on, it
  # will start again at the top of the next. For example, it
  # will start at w_12, work its way down to w_(N-1)(N), and
  # then the if statement will move it back up to w_13
  # whereupon it will work its way down to w_(N-2)(N), and
  # so on.
  i <- 1
  j <- 2
  while (j <= N) {
    
    # we define the weigts with the hebb rule (where p is the
    # pattern matrix which is N by P), and then ensure that
    # our weight matrix is symetric.
    weights[i,j] <<- 1/N * (patterns[i,] %*% patterns[j,])
    weights[j,i] <<- weights[i,j]
    
    i <- i+1
    j <- j+1
    
    if(j>N){
      j <- 3+(N-i)
      i <- 1
    }
  }
  
  # update the activation vector
  act.vector <<- as.vector(weights %*% network)
  
  # update the stay.chance.vector (by using the one.chance.vector
  # as an intermediary step)
  one.chance.vector <- sapply(act.vector, act.to.prob)
  stay.chance.vector <<- mapply(function(state, prob) {
    if (state == 1) {
      return(prob)
    } else{
      return(1 - prob)
    }
  }, network, one.chance.vector)
}

# This function initializes the network of a specific size and temurature, and then
# trains it to retrieve N*alpha patterns
initialze.network <- function(num.neurons, alpha, temp) {
  
  # Number of neurons
  N <<- num.neurons
  
  # Number of patterns
  P <<- floor(num.neurons*alpha)
  
  # Temperature
  T <<- temp
  
  # Alpha
  alpha <<- P/N
  
  
  network <<- rep(0, N)
  
  randomize.patterns()
  
  simple.train()
}



# Visualizing the Network -------------------------------------------------


# This function plots a visualization of teh network. Neurons in a 1 state
# are represented by black circles, while neurons in the -1 state are
# represented by white circles. The activation of the neuron is visualized
# as a red or blue circle surrounding it representing positive and negative
# activation respectively. The color intensity scales with the level of
# activation.
plot.network <- function(){
  
  # function for activation color.
  act.color <- function(act){
    # Note that the theoretical maximum activation with trained weights given this training method is P,
    # but in practice the activations rarely ecxeed 1, so we set 1 as the maximum color intensity. We also
    # set a minimum intensity for nonzero activations to make all the signs of the activations visible
    
    # creates color spectra
    blue.scale <- colorRamp(c(rgb(190, 190, 255, maxColorValue = 255), "blue"))
    red.scale <- colorRamp(c(rgb(255, 190, 190, maxColorValue = 255), "red"))
    
    
    # This function ensures that if the activation exceeds 1 in absolute value, it will still be represented
    # as the maximum intensity in the color spectra.
    stabilize <- function(x) {
      if(x > 1){
        x <- 1
      }
      if(x < -1){
        x <- -1
      }
      return(x)
    }
    
    # blue represents positive activation, red negative, and white zero
    if(act > 0){
      return(rgb(blue.scale(stabilize(act)), maxColorValue = 255))
    }
    if(act < 0){
      return(rgb(red.scale(-stabilize(act)), maxColorValue = 255))
    }
    else{
      return("white")
    }
  }
  
  # makes a blank plot
  plot(0, asp = 1,  axes=FALSE, ann=FALSE, type="n", xlab="", ylab="", xlim=c(0, sqrt(N)+1), ylim=c(0, sqrt(N)+1))
  # define circle radii
  neuron.radius <- 0.32
  act.radius <- 0.45
  
  # plot the circles in a square grid
  for(x in 1:sqrt(N)){
    for (y in 1:sqrt(N)){
      # this means we stack the state vecton in rows of sqrt(N)
      # length to get the square (so this only works correctly when
      # N is a square number)
      pos.value <- x + (y-1) * sqrt(N)
      draw.circle(x, y, act.radius, col = act.color(act.vector[pos.value]))
      draw.circle(x, y, neuron.radius, col = (network[pos.value]+1)/2)
    }
  }
}





# Updating the Network ----------------------------------------------------


# This function runs the network through n updates. If graph is set to true,
# then the function will plot the network once at the start, once after every
# graph.freq successful updates (i.e. updates that esulted in a change in the
# state of a neuron), and once after the final update (if there has been a
# change in the network state since the last graph).
update.ntimes <- function(n ,graph, graph.freq) {
  
  # This makes graph and as a result graph.freq optional arguments
  if(missing(graph)){
    graph <- FALSE
  }
  
  # plot the original network
  if(graph){
    plot.network()
  }
  
  # counter to keep track of number of succesful updates
  num.changes <- 0
  
  # update the network n times
  for(i in 1:n) {
    
    # select random neuron
    num <-sample(1:N, 1)
    
    # find the new state and the updated state
    act <- act.vector[num]
    old.state <- network[num]
    new.state <- act.to.state(act)
    
    # if the states are different, change the neuron to its new state,
    # update the activation vector and increment num.changes.
    if(new.state != old.state){
      
      network[num] <<- new.state
      act.vector <<- as.vector(weights %*% network)
 
      num.changes <- num.changes + 1
      
      # plot the network if there have been another graph.freq
      # successful updates and graph = TRUE
      if(graph && (num.changes+1)%%graph.freq == 0){
        plot.network()
      }
    }
  }
  
  # If there have been more successful updates since the last plot
  # and graph = TRUE, then plot the network
  if(graph){
    if(num.changes%%graph.freq != 0){
      plot.network()
    }
    
    # return num.changes the network underwent
    print(num.changes)
  }
  
  # update the stay.chance.vector (by using the one.chance.vector
  # as an intermediary step)
  one.chance.vector <- sapply(act.vector, act.to.prob)
  stay.chance.vector <<- mapply(function(state, prob) {
    if (state == 1) {
      return(prob)
    } else{
      return(1 - prob)
    }
  }, network, one.chance.vector)
}





# Naive Approach to Stability and the Phase Diagram -----------------------


naive.is.stable <- function(stab.threshold) {
  # define a vector that represents the chance that each neuron
  # will be a 1 after its next update
  one.chance.vector <- sapply(act.vector, function(act) {
    return(1 / (1 + exp(-2 * act / T)))
  })
  
  stay.chance.vector <- mapply(function(state, prob) {
    if (state == 1) {
      return(prob)
    } else{
      return(1 - prob)
    }
  }
  , network, one.chance.vector)
  
  return(min(stay.chance.vector) >= stab.threshold)
}

naive.test.pattern.stability <- function(sample.size, stab.threshold) {

  errors <- 0

  if(sample.size > P){
    sample.size <- P
  }

  for(i in 1:sample.size){
    network <<- patterns[,i]
    act.vector <<- as.vector(weights%*%network)
    if(!naive.is.stable(stab.threshold)){
      errors <- errors+1
    }
  }

  randomize.network()

  network.error <- 1-errors/sample.size

  return(network.error)
}

naive.average.pattern.stability <- function(num.neurons, alpha, temp, stab.threshold, num.trials, sample.size){

  counter <- 0

  for (i in 1: num.trials) {
    initialze.network(num.neurons, alpha, temp)
    counter <- counter + naive.test.pattern.stability(sample.size, stab.threshold)
  }

  return(counter/num.trials)
}

naive.plot.phase.diagram <- function(t.step, a.step, num.neurons, stab.threshold, num.trials, sample.size) {
  # # creates color spectra
  # blue.scale <- colorRamp(c(rgb(100, 100, 100, maxColorValue = 255), "blue"))
  # red.scale <- colorRamp(c(rgb(100, 100, 100, maxColorValue = 255), "red"))
  # green.scale <- colorRamp(c(rgb(100, 100, 100, maxColorValue = 255), "green"))

  num.a.steps <- floor(0.16/a.step)
  num.t.steps <- floor(1/t.step)

  plot(c(a.step,a.step,0.16,0.16), c(t.step,1,t.step,1), pch = NA, xlab = "alpha",ylab = "T", main = "Stored Pattern Stability")

  point.matrix <- data.frame("alpha" = NULL, "T" = NULL, proportion.stable <- NULL)

  for (i in 1:num.a.steps) {
    for (j in 1:num.t.steps) {
      p.alpha <- a.step*i
      p.temp <- t.step*j
      
      # this if statement is to save computation time by not calculating average stability for some points
      if(5*p.alpha + p.temp <= 1.1){
      
      p.stability <- naive.average.pattern.stability(num.neurons, p.alpha, p.temp, stab.threshold, num.trials, sample.size)

      point.matrix <- rbind(point.matrix, c(p.alpha, p.temp, p.stability))

      # total.stability <-stability^(alpha*num.neurons)

      if(p.stability >= 0.99){
        type <- 16
        color <- grey(1-p.stability)
      } else{
        type <- 15
        color <- grey(1-p.stability)
      }

      points(p.alpha, p.temp, pch = type, col = color)
      }
    }
  }

  return(point.matrix)
  
}

phase.diagarm.data <- naive.plot.phase.diagram(0.05, 0.005, 500, 0.9, 100, 20)





# Network Evolution Data --------------------------------------------------


# This function provides data describing the behavior of the network as it evolves.
# It tracks the hamming distance and energy difference between the network and another
# designated vector. num.updates controls how long the evolution that's being tested is,
# and state.vector controls what state the network is being compared to as it evolves
# (the default for this argument is network so that it compares itself to its original
# state, but if we expect the network to evolve towards a known pattern we can instead
# track its movement in comparison with that state)
evolution.data <- function(num.updates, state.vector) {
  
  # this state.vector an optional argument which defaults to the current network state
  if(missing(state.vector)){
    state.vector <- network
  }

  # define the original state of the network
  original.state <- network
  
  # represents the energy difference between the network and the state.vector
  delta.energy <- energy(network) - energy(state.vector)
  
  # vector to track the energy difference between the network and the state.vector
  # after each update
  delta.energy.vec <- delta.energy
  
  # vector to track the hamming distance between the network and the state.vector
  # after each update
  hamming.dist.vec <- hamming(state.vector)
  
  # update the network num.updates times and build the two vectors
  for(i in 1:num.updates) {
    
    # select random neuron
    num <-sample(1:N, 1)
    
    # find the new state and the updated state
    act <- act.vector[num]
    old.state <- network[num]
    new.state <- act.to.state(act)
    
    # if the states are different, change the neuron to its new state,
    # update the activation vector, and update delta.energy.
    if(new.state != old.state){
      
      network[num] <<- new.state
      act.vector <<- as.vector(weights %*% network)
      
      # this uses the energy.change function written above to calculate the new
      # enegy difference without using the more computationally intensive energy
      # function.
      delta.energy <- delta.energy + energy.change(num, old.state, new.state)
    }
    
    # add the new hamming distance and energy difference to the end of their
    # respective vectors
    delta.energy.vec <- c(delta.energy.vec, delta.energy)
    hamming.dist.vec <- c(hamming.dist.vec, hamming(state.vector))
    
  }
  
  # set the network and activation vector back to their original state
  network <<- original.state
  act.vector <<- as.vector(weights %*% network)
  
  # create a data frame that holds both that hamming and energy data vectors
  data <- data.frame(hamming = hamming.dist.vec, energy = delta.energy.vec)
  
  # output the data frame
  return(data)
}





# Basin of Attraction -----------------------------------------------------


# corrupt the a pattern by a given percentage
corrupt <- function(pattern, percent) {
  
  # make corrupted pattern
  corrupted.pattern <- pattern
  
  # select a given perventage of the neurons randomly
  len <- length(pattern)
  num.changes <- round(len*percent/100)
  selected.neurons <- sample(1:len, num.changes, replace = FALSE)
  
  # flip the value of each selected neuron
  for (i in 1:num.changes) {
    index <- selected.neurons[i]
    
    # this switches the value of the neuron
    corrupted.pattern[index] <- ((corrupted.pattern[index] +3)%%4 - 1)
  }
  
  # return the corrupted pattern
  return(corrupted.pattern)
}





# Stability Measure -------------------------------------------------------


# This function takes in a data frame which describes the change in hamming distance
# and energy undergone by a network over the course of its evolution. It then outputs
# a new data frame containing 7 mertics that summarize the raw evolution data. These
# metrics are chosen to help classify the network state as stable or unstable based on
# its behavior upon prolonged evolution
behavior.summary <- function(hamming.energy.data) {
  
  # extract the hamming dist and energy vectors form the data frame
  hamming.vec <- hamming.energy.data$hamming
  energy.vec <- hamming.energy.data$energy
  
  # label the number of updates the evolution data represents
  test.length <- length(hamming.vec) - 1
  
  # we ask if the network stays within an energy basin of which
  # the initial state is the minimum.
  
  
  # In particular, we want to understand the fluctuations in the network, i.e.
  # the periods when the network diverges from the lowest energy stable state
  # but still remains within the surrounding energy basin. I have chosen seven
  # metrics that can be extracted from the raw hamming distance evolution data
  # the set of which give a good understanding of how the network evolves. The
  # metrics are as follows:
  
  # The average hamming distance from the original state
  # The average size of the fluctuations in h-dist
  # The maximim duration of the fluctutions in h-dist
  # The proportion of time spent not in the original state
  # Whether the final fluctuation is exceedingly long lived both absolutely and relatively
  # Whether the final fluctuation is an exceedinly tall both absolutely and relatively
  # Whether the network stays within an energy basin of which it is the minimum
  
  # We create 2 vectors represtenting the fluctuation duration and the fluctuation size.
  fluctuation.time <- NULL
  fluctuation.size <- NULL
  
  # within.basin is true if the network is does not evolve to an energy state lower than
  # the one it began in
  within.basin <- (round(min(energy.vec), digits=5) == 0)
  
  # We use two counters to construct the above vectors in a for loop over the hamming-dist data
  fluctuation.counter <- NULL
  time.counter <-0
  for (i in 2:(1+test.length)) {
    
    # label the current and previous states in the hamming distance vector
    state <- hamming.vec[i]
    prev.state <- hamming.vec[i-1]
    
    # We test 4 conditions to stack when the network is stable or fluctuating 
    if(state == 0 && prev.state == 0){
      # this code only needs to run if we want to keep track of the duration of
      # periods of stability in the evolution
        # time.counter <- time.counter+1
        # 
        # if(i == 1 + test.length){
        #   stable.time <- c(stable.time, time.counter)
        # }
    }
    if(state != 0 && prev.state == 0){
      # we add the hamming dist to the end of the fluctuation size counter
      fluctuation.counter <- c(fluctuation.counter, state)
      
      # this code only needs to run if we want to keep track of the duration of
      # periods of stability in the evolution
        # stable.time <- c(stable.time, time.counter)
        # time.counter <- 0
    }
    if(state != 0 && prev.state != 0){
      # add to the duration and fluctuation size counter
      time.counter <- time.counter+1
      fluctuation.counter <- c(fluctuation.counter, hamming.vec[i])
      
      # if the data ends in a fluctuation, we store the fluctuation duration and size
      # in their respective vectors, and store the duration of the final fluctuation
      if(i == 1 + test.length){
        fluctuation.time <- c(fluctuation.time, time.counter)
        fluctuation.size <- c(fluctuation.size, max(fluctuation.counter))
      }
    }
    if(state == 0 && prev.state != 0){
      # we store the fluctuation duration and size in their respective vectors
      fluctuation.time <- c(fluctuation.time, time.counter)
      fluctuation.size <- c(fluctuation.size, max(fluctuation.counter))
      
      # we reset the counters
      time.counter <- 0
      fluctuation.counter <- NULL
    }
  }
  
  
  
  # We create the data frame of the 7 metrics of stablility
  behavior.summary <- data.frame(
    
    ave.dist = mean(hamming.vec),
    
    ave.flux.size = mean(fluctuation.size),
    
    max.flux.time = max(fluctuation.time),
    
    prop.time.unstable = 1 - length(hamming.vec[x = 0])/test.length,
    
    final.is.long = (tail(fluctuation.time) >= 2*max(fluctuation.time)) &&
      (tail(fluctuation.time) >= 50*N),
    
    final.is.large = (tail(fluctuation.size) >= 2*max(fluctuation.size)) &&
      (tail(fluctuation.size) >= 0.1),
    
    within.basin = within.basin
    
    # these are some additional metrics that I found insufficiently informative
      # ave.flux.time = mean(fluctuation.time),
      # num.flux = length(fluctuation.time),
      # max.stab.time = max(stable.time),
      # ave.stab.time = mean(stable.time),
      # max.flux.size = max(fluctuation.size),
  )
  
  # return the summary data frame
  return(behavior.summary)
}

# This function takes in the data frame summarizing the network evolution, and outputs
# a number between 0 and 1 representing the level of stability of the network state
stable.1 <- function(behavior.summary) {
  
  # curve roughly corresponds to the rate of dropoff in the stability measure as the
  # network diverges from the maximally stable point. The metric becomes astep funcion
  # as curve approacked infinity
  curve <- 3.2
  
  # amp determined the maximum possible value of the function
  amp <- 1
  
  # we extract the metrics form the date frame
  ave.dist <- behavior.summary$ave.dist
  ave.flux.size <- behavior.summary$ave.flux.size
  max.flux.time <- behavior.summary$max.flux.time
  prop.time.unstable <- behavior.summary$prop.time.unstable
  
  # these two if conditions ensure that the function outputs a 1 in the case where
  # there are no fluctuations whatsoever in the network evolution
  if(is.na(ave.flux.size)){
    ave.flux.size <- 0
  }
  if(max.flux.time == -Inf){
    max.flux.time <- 0
  }
  
  # these thresholds represent the coordinates of the outermost point that we still
  # consider to be stable. This point sits on the boundary of a superellipsoid.
  # Note that the network can still have nonzero stability if it is past one of these
  # thresholds but behind another. To find the absolute thresholds for each of these
  # values, multuply them all by 4^(1/curve).
  ave.dist.th <- 0.01
  ave.flux.size.th <- 0.02
  prop.time.unstable.th <- 0.96
  max.flux.time.th <- 25*N
  
  # this statement tests whether the final fluctuation was exceedingly long or large. If yes,
  # then we set the amplitude to 0
  if(!is.na(behavior.summary$final.is.long)){
    if(behavior.summary$final.is.long || behavior.summary$final.is.large){
      amp <- 0
    }
  }
  
  # this tests whether the network remained the minimum of the energy basin it occupied.
  # if no, then we reduce the stability measure by setting amp to 0.75
  if(!behavior.summary$within.basin){
    amp <- 0.75
  }
  
  # these statements define the stability fucntion as a piecewise fucntion which is either a section of
  # a super-hyperellipsoid embedded in R5, or an R4 plane at stability = 0
  value <- 1 - 0.25*(ave.dist/ave.dist.th)^curve - 0.25*(ave.flux.size/ave.flux.size.th)^curve
  - 0.25*(max.flux.time/max.flux.time.th)^curve - 0.25*(prop.time.unstable/prop.time.unstable.th)^curve
  
  if( value > 0){ 
    stability <- amp*value^(1/curve)
  }
  else{
    stability <- 0
  }
  
  # output the level of stability
  return(stability)
}

# this function updates the network num.updates times, and uses that evolution data to
# determine the stability of the network state. If graph = TRUE it will also graph the data.
stability.test <- function(num.updates, graph) {
  
  # this makes graph an optional argument
  if(missing(graph)){
    graph <- FALSE
  }
  
  # We collect the network evolution data
  data <- evolution.data(num.updates)
  hamming.vec <- data$hamming
  energy.vec <- data$energy
  
  # use behavior.summary to summarize the network behavior
  summary <- behavior.summary(data)
  
  # measure the stability of the network state
  stability <- stable.1(summary)
  
  # graph the network elvolution and label it with the assigned stability number if graph = TRUE
  if(graph){
    plot(0:num.updates, energy.vec, type = "l",
         xlab = "Time", ylab = "Energy Difference from Original State", main = paste("stability =", stability))
    
    plot(0:num.updates, hamming.vec,  type = "l",
         xlab = "Time", ylab = "Distance from Original State", main = paste("stability =", stability))
  }
  
  # output the stability measure of the network state
  return(stability)
}



# this function generates a set of stability test graphs that allow us to see how well
# our stability metric tracks our intuition about what stability is
test.stability.measure <- function() {
  # run stability tests for alpha in 15/200 to 35/200 at T = 0.2
  for (i in 15:35) {
    initialze.network(200, i/200, 0.2)
    
    # test the stability of 4 of the stored patterns
    for (j in 1:4) {
      set.as.pattern(j)
      
      stability.test(150*N, graph = TRUE)
    }
  }
}



# Predicting Network Stability --------------------------------------------

# this function gives a summary of the superficial characteristics of the network
# state which we will use to try to predict the network's stability level
state.summary <- function() {
  
  vector.low <- sort(stay.chance.vector)[1:(length(stay.chance.vector)/10)]
  
  all.data <- boxplot.stats(stay.chance.vector)$stats
  
  low.data <- boxplot.stats(vector.low)$stats
  
  data <- as.data.frame(cbind(
    min = min(stay.chance.vector),
    low.q1 = low.data[2],
    low.median = low.data[3],
    low.q2 = low.data[4],
    all.q1 = all.data[2],
    all.median = all.data[3],
    all.q2 = all.data[4],
    max = max(stay.chance.vector),
    alph = alpha,
    temp = T,
    full.vector.ordered = list(sort(stay.chance.vector)) 
  ))
  
  return(data)
}

# this function creates a data set that pairs the network state summary with
# its actual stability level for 510 different states. This data will be used to
# out what characteristics of a state best predict its stability
correlation.data <- function() {
  
  data <- data.frame(
    stability = NULL,
    min = NULL,
    low.q1 = NULL,
    low.median = NULL,
    low.q2 = NULL,
    all.q1 = NULL,
    all.median = NULL,
    all.q2 = NULL,
    max = NULL,
    alph = NULL,
    temp = NULL,
    full.vector = NULL
    
  )
  
  for (t in 1:15){
    # this takes T from 0.05 to 0.75
    for (i in 1:17) {
      # this takes alpha from 0.01 to 0.17
      
      initialze.network(200, i*2/200, 0.05*t)
      
      for (j in 1:2) {
        set.as.pattern(min(c(j, 2*i)))
        
        stability <- stability.test(150*N)
        
        data.point <- state.summary()
        data.point$stability = stability
        
        data <- rbind(data, data.point)
      }
    }
  }
  
  return(data)
}


# This code gives us a data frame that we can use nls on
stability.correlation.data <- correlation.data()
simpler.stability.correlation.data <- stability.correlation.data[, c(1:10, 12)]
myFun <- function(data) {
  ListCols <- sapply(data, is.list)
  cbind(data[!ListCols], t(apply(data[ListCols], 1, unlist)))
}
simpler.stability.correlation.data <- myFun(simpler.stability.correlation.data)


# This implements linear regression
model <- nls(
  stability ~ A* min + B*low.q1 + C*low.median + D*low.q2 + E*all.q1 + F*all.median + G*all.q2 + H*max + I,
  data = simpler.stability.correlation.data,
  start = c(A = 0., B = 0.1, C = 0.1, D = 0.1, E = 0.1, F = 0.1, G = 0.1, H = 0.1, I = 0.1),
  control = nls.control(maxiter = 1000, minFactor = 0, warnOnly = TRUE)
)

summary(model)


plot(nlsResiduals(model), which = 1)
plot(1:510, residuals(model))


# This fits the data to a sigmoid function
model <- nls(
  stability ~ 1/(1+exp(-1*(A*min + B*low.q1 + C*low.median + D*low.q2 + E*all.q1 + F*all.median + G*all.q2 + H*max + I))),
  data = simpler.stability.correlation.data,
  start = c(A = 0., B = 0.1, C = 0.1, D = 0.1, E = 0.1, F = 0.1, G = 0.1, H = 0.1, I = 0.1),
  control = nls.control(maxiter = 10000, minFactor = 0, warnOnly = TRUE), trace = TRUE
)

summary(model)



# This fits the data to a sigmoid function and removes extra data that contributes to overfitting
model <- nls(
  stability ~ 1/(1+exp(-1*(A*min + B*low.q1 + D*low.q2 + E*all.q1 + G*all.q2 + H*max))),
  data = simpler.stability.correlation.data,
  start = c(A = 0., B = 0.1, D = 0.1, E = 0.2, G = 0.2, H = 0.2),
  control = nls.control(maxiter = 10000, minFactor = 0, warnOnly = TRUE), trace = TRUE
)

summary(model)



plot(nlsResiduals(model), which = 1)
plot(1:510, residuals(model))



# This fits the model to a sigmiod fucntion with onlt the minimum stay chance as an input
model <- nls(
  stability ~ 1/((1+exp(-B*(min - A))))
  #*(1+exp(-(a*min + b*low.q1 + c*low.median + d*low.q2 + e*all.q1 + f*all.median + g*all.q2 + h*max + i)))
  ,
  data = simpler.stability.correlation.data,
  start = c(A = 0.93, B = 1000),
  control = nls.control(maxiter = 10000, minFactor = 0, warnOnly = TRUE), trace = TRUE
)

summary(model)


plot(nlsResiduals(model), which = 1)
plot(1:510, residuals(model))






plot(1:510, residuals(model), )






# Naive Approach to Stability and the Phase Diagram -----------------------


naive.is.stable <- function(stab.threshold) {
  # define a vector that represents the chance that each neuron
  # will be a 1 after its next update
  one.chance.vector <- sapply(act.vector, function(act) {
    return(1 / (1 + exp(-2 * act / T)))
  })
  
  stay.chance.vector <- mapply(function(state, prob) {
    if (state == 1) {
      return(prob)
    } else{
      return(1 - prob)
    }
  }
  , network, one.chance.vector)
  
  return(min(stay.chance.vector) >= stab.threshold)
}

naive.test.pattern.stability <- function(sample.size, stab.threshold) {
  
  errors <- 0
  
  if(sample.size > P){
    sample.size <- P
  }
  
  for(i in 1:sample.size){
    network <<- patterns[,i]
    act.vector <<- as.vector(weights%*%network)
    if(!naive.is.stable(stab.threshold)){
      errors <- errors+1
    }
  }
  
  randomize.network()
  
  network.error <- 1-errors/sample.size
  
  return(network.error)
}

naive.average.pattern.stability <- function(num.neurons, alpha, temp, stab.threshold, num.trials, sample.size){
  
  counter <- 0
  
  for (i in 1: num.trials) {
    initialze.network(num.neurons, alpha, temp)
    counter <- counter + naive.test.pattern.stability(sample.size, stab.threshold)
  }
  
  return(counter/num.trials)
}

naive.plot.phase.diagram <- function(t.step, a.step, num.neurons, stab.threshold, num.trials, sample.size) {
  # # creates color spectra
  # blue.scale <- colorRamp(c(rgb(100, 100, 100, maxColorValue = 255), "blue"))
  # red.scale <- colorRamp(c(rgb(100, 100, 100, maxColorValue = 255), "red"))
  # green.scale <- colorRamp(c(rgb(100, 100, 100, maxColorValue = 255), "green"))
  
  num.a.steps <- floor(0.16/a.step)
  num.t.steps <- floor(1/t.step)
  
  plot(c(a.step,a.step,0.16,0.16), c(t.step,1,t.step,1), pch = NA, xlab = "alpha",ylab = "T")
  
  point.matrix <- matrix(nrow = 0, ncol = 3)
  
  for (i in 1:num.a.steps) {
    for (j in 1:num.t.steps) {
      p.alpha <- a.step*i
      p.temp <- t.step*j
      p.stability <- naive.average.pattern.stability(num.neurons, p.alpha, p.temp, stab.threshold, num.trials, sample.size)
      
      point.matrix <- rbind(point.matrix, c(p.alpha, p.temp, p.stability))
      
      # total.stability <-stability^(alpha*num.neurons)
      
      if(p.stability >= 0.99){
        type <- 16
        color <- grey(1-p.stability)
      } else{
        type <- 15
        color <- grey(1-p.stability)
      }
      
      points(p.alpha, p.temp, pch = type, col = color)
    }
  }
  
  stability.data <<- point.matrix
}



# Original Zero Tempurature implemetation ---------------------------------

# 
# # BASE NETWORK
# 
# # Number of neurons
# N <- 1
# # Number of patterns
# P <- 1
# # Alpha
# alpha <- P/N
# 
# # Blank network
# network <- rep(0, N)
# # Blank patterns
# patterns <- matrix(data = rep(0, N), nrow = N, ncol = P)
# # Blank weights
# weights <- matrix(data = rep(0, N^2), nrow = N, ncol = N)
# # Blank activation vector
# act.vector <- rep(0, N)
# 
# 
# 
# initialze.network <- function(num.neurons, alpha) {
#   
#   # Number of neurons
#   N <<- num.neurons
#   # Number of patterns
#   P <<- floor(num.neurons*alpha)
#   # Alpha
#   alpha <<- P/N
#   
#   
#   network <<- rep(0, N)
#   
#   randomize.patterns()
#   
#   weights <<- matrix(, nrow = N, ncol = N)
#   
#   simple.train()
# }
# 
# 
# # Create random state vector
# random.state.vector <- function() {
#   return(sample(c(-1,1), N, replace = TRUE))
# }
# 
# 
# 
# # randomize the network
# randomize.network <- function(){
#   network <<- random.state.vector()
#   act.vector <<- as.vector(weights %*% network)
# }
# 
# # Create pattern set
# randomize.patterns <- function(){
#   patterns <<- replicate(P, random.state.vector())
# }
# 
# # Training weights
# simple.train <- function(){
#   # set the weight matrix diagonals to zero
#   for (d in 1:N) {
#     weights[d,d] <<- 0
#   }
#   
#   # make a whle loop that covers the upper triangle of the
#   # weight matrix, moving diagonally downwards starting at
#   # w_12. The if statement at the bottom ensures that when
#   # the loop reaches the bottom of the diagonal it is on, it
#   # will start again at the top of the next. For example, it
#   # will start at w_12, work its way down to w_(N-1)(N), and
#   # then the if statement will move it back up to w_13
#   # whereupon it will work its way down to w_(N-2)(N), and
#   # so on.
#   i <- 1
#   j <- 2
#   while (j <= N) {
#     
#     # we define the weigts with the hebb rule (where p is the
#     # pattern matrix which is N by P), and then ensure that
#     # our weight matrix is symetric.
#     weights[i,j] <<- 1/N * (patterns[i,] %*% patterns[j,])
#     weights[j,i] <<- weights[i,j]
#     
#     i <- i+1
#     j <- j+1
#     
#     if(j>N){
#       j <- 3+(N-i)
#       i <- 1
#     }
#   }
#   act.vector <<- as.vector(weights %*% network)
# }
# 
# 
# 
# 
# 
# 
# # Ploting the resulting neural network
# 
# 
# 
# act.color <- function(act){
#   # creates color spectra
#   blue.scale <- colorRamp(c(rgb(180, 180, 255, maxColorValue = 255), "blue"))
#   red.scale <- colorRamp(c(rgb(255, 180, 180, maxColorValue = 255), "red"))
#   
#   stabilize <- function(x) {
#     if(x > 1){
#       x <- 1
#     }
#     if(x < -1){
#       x <- -1
#     }
#     return(x)
#   }
#   
#   if(act > 0){
#     return(rgb(blue.scale(stabilize(act)), maxColorValue = 255))
#   }
#   if(act < 0){
#     return(rgb(red.scale(-stabilize(act)), maxColorValue = 255))
#   }
#   else{
#     return("white")
#   }
# }
# 
# 
# # visualizing activateion with black vs white neurons
# 
# plot.network <- function(state.vector){
#   
#   # makes a blank plot
#   plot(0, asp = 1,  axes=FALSE, ann=FALSE, type="n", xlab="", ylab="", xlim=c(0, sqrt(N)+1), ylim=c(0, sqrt(N)+1))
#   # define circle radii
#   neuron.radius <- 0.32
#   act.radius <- 0.45
#   
#   # define the activation vector
#   act.vector <- weights %*% state.vector
#   
#   # plot the circles in a square grid
#   for(x in 1:sqrt(N)){
#     for (y in 1:sqrt(N)){
#       # this means we stack the state vecton in rows of sqrt(N)
#       # length to get the square
#       pos.value <- x + (y-1) * sqrt(N)
#       draw.circle(x, y, act.radius, col = act.color(act.vector[pos.value]))
#       draw.circle(x, y, neuron.radius, col = (state.vector[pos.value]+1)/2)
#     }
#   }
# }
# 
# 
# 
# # network update fucntion
# 
# # neuron activation to neuron state function
# act.to.state <- function(act){
#   if(act > 0){
#     return(1)
#   }
#   if(act < 0){
#     return(-1)
#   } else{
#     return(0)
#   }
# }
# 
# 
# 
# # chach if the network is a pattern
# is.pattern <- function(){
#   
#   output <- FALSE
#   
#   # check if the network is identical to any of the patterns
#   for (i in 1:P) {
#     
#     # if it is for any pattern, set the output to true and
#     # end the for loop
#     if(identical(network, patterns[,i])){
#       output <- TRUE
#       i <- P+1
#     }
#   }
#   
#   # return the result
#   return(output)
# }
# 
# 
# 
# # corrupt the a pattern by a given percentage
# 
# corrupt <- function(pattern, percent) {
#   
#   # make corrupted pattern
#   corrupted.pattern <- pattern
#   
#   # select a given perventage of the neurons randomly
#   len <- length(pattern)
#   num.changes <- round(len*percent/100)
#   selected.neurons <- sample(1:len, num.changes, replace = FALSE)
#   
#   # flip the value of each selected neuron
#   for (i in 1:num.changes) {
#     index <- selected.neurons[i]
#     
#     # this switches the value of the neuron
#     corrupted.pattern[index] <- ((corrupted.pattern[index] +3)%%4 - 1)
#   }
#   
#   # return the corrupted pattern
#   return(corrupted.pattern)
# }
# 
# 
# 
# 
# 
# # Check if the network is in a stable state
# 
# is.stable <- function(){
#   
#   # define a vector that represents all possible updates
#   possibility.vector <- sapply(act.vector, act.to.state)
#   
#   # return true if there is no zero in the sum of the network
#   # and its posibility vector
#   return(!(0 %in% (network + possibility.vector)))
# }
# 
# 
# # Let network evolve to a stable state
# 
# evolve <- function(graph, graph.freq) {
#   
#   if(graph){
#     plot.network(network)
#   }
#   
#   # Define a variable to keep track of the number of
#   # neuron updates, changes, and consecutive failed
#   # updates that occur
#   num.updates <- 0
#   num.changes <- 0
#   consec.failed.updates <- 0
#   
#   # define the while loop condition
#   condition <- TRUE
#   
#   # update the network n times
#   while (condition) {
#     
#     # increment num.updates
#     num.updates <- num.updates +1
#     
#     # select random neuron
#     num <-sample(1:N, 1)
#     
#     # find the new state and the updated state
#     act <- act.vector[num]
#     old.state <- network[num]
#     new.state <- act.to.state(act)
#     
#     # if the activation is nonzero and the states are different
#     # change the neuron to its new state, update the activation
#     # vector, increment num.changes, and set consec.failed,updates
#     # to zero
#     if(new.state != 0 && new.state != old.state){
#       
#       network[num] <<- new.state
#       act.vector <<- as.vector(weights %*% network)
#       
#       if(graph && (num.changes+1)%%graph.freq == 0){
#         plot.network(network)
#       }
#       
#       num.changes <- num.changes + 1
#       consec.failed.updates <- 0
#     }
#     
#     # otherwise count it as a failed update
#     else{
#       consec.failed.updates <- consec.failed.updates + 1
#     }
#     
#     # there's about a 4% cahnce that some neuron has not
#     # been updated is after 9N updates. This number changes very slightly with N.
#     # the derivation is at https://www.desmos.com/calculator/dyomwdesru.
#     if(consec.failed.updates > 9*N && is.stable()){
#       condition <- FALSE
#       
#       num.updates <- num.updates - consec.failed.updates
#     }
#   }
#   
#   if(graph){
#     plot.network(network)
#     
#     # return num.cahnges and num.updates required to stabalize the network
#     print(c(num.changes, num.updates))
#   }
# }
# 
# 
# 
# # Test pattern retrieval accuracy
# 
# test.retrieval <- function(num.tests, corruption.percent) {
#   
#   errors <- 0
#   
#   for(i in 1:P){
#     for(j in 1:num.tests){
#       network <<- corrupt(patterns[,i], corruption.percent)
#       act.vector <<- as.vector(weights %*% network)
#       evolve(FALSE)
#       
#       if(!is.pattern()){
#         errors <- errors+1
#       }
#     }
#   }
#   
#   network.error <- 1-errors/(P*num.tests)
#   ave.pattern.error <- (network.error)^(1/N)
#   
#   return(c(ave.pattern.error, network.error))
# }
# 
# test.pattern.stability <- function(sample.size) {
#   
#   errors <- 0
#   
#   if(sample.size > P){
#     sample.size <- P
#   }
#   
#   for(i in 1:sample.size){
#     network <<- patterns[,i]
#     act.vector <<- as.vector(weights%*%network)
#     if(!is.stable()){
#       errors <- errors+1
#     }
#   }
#   
#   randomize.network()
#   
#   network.error <- 1-errors/sample.size
#   
#   return(network.error)
# }
# 
# 
# 
# average.pattern.stability <- function(num.neurons, alpha, num.trials, sample.size){
#   
#   counter <- 0
#   
#   for (i in 1: num.trials) {
#     initialze.network(num.neurons, alpha)
#     counter <- counter + test.pattern.stability(sample.size)
#   }
#   
#   return(counter/num.trials)
# }
# 
# average.pattern.stability(225, 0.05, 100, 100)
# 
# 
# 
# initialze.network(144, .1)
# randomize.network()
# 
# evolve(TRUE, 10)




# Scrapped Code -----------------------------------------------------------


# 
# curve <- 6
# amp <- 1
# 
# ave.dist <- behavior.summary$ave.dist
# ave.flux.size <- behavior.summary$ave.flux.size
# max.flux.time <- behavior.summary$max.flux.time
# prop.time.unstable <- behavior.summary$prop.time.unstable
# 
# ave.dist.th <- 0.015
# ave.flux.size.th <- 0.025
# prop.time.unstable.th <- 0.96
# max.flux.time.th <- 30*N
# 
# if(!is.na(behavior.summary$final.flux.time) && behavior.summary$final.flux.time >= 60*N){
#   amp <- 0
# }
# if(!behavior.summary$within.basin){
#   amp <- 0.75
# }
# 
# t1 <- (ave.dist/ave.dist.th)^curve
# t2 <- (ave.flux.size/ave.flux.size.th)^curve
# t3 <- (max.flux.time/max.flux.time.th)^curve
# t4 <- (prop.time.unstable/prop.time.unstable.th)^curve
# 
# stability <- amp*(1 - t1 - t2 - t3 - t4)^(1/curve)
# 
# print(t1)
# print(t2)
# print(t3)
# print(t4)
# print(stability)




# WHAT COUNTS AS STABLE?

# 
# stable.states <- data.frame(mean = NULL, min = NULL)
# 
# initialze.network(400, 0.1, 0.1)
# 
# for (i in 1:P) {
#   
#   set.as.pattern(i)
#   
#   one.chance.vector <- sapply(act.vector, function(act) {
#     return(1 / (1 + exp(-2 * act / T)))
#   })
#   
#   stay.chance.vector <- mapply(function(state, prob){
#     if (state == 1) {
#       return(prob)
#     } else{
#       return(1 - prob)
#     }
#   }
#   , network, one.chance.vector)
#   
#   stay.chance.mean.min <- c(mean(stay.chance.vector), min(stay.chance.vector))
#   
#   
#   
#   update.ntimes(stability.measure, FALSE)
#   
#   if(hamming(patterns[,i] <= 0.02))
# }
# 





# 
# 
# # Let network evolve to a stable state
# 
# evolve <- function(graph, graph.freq, stab.threshold) {
#   
#   if(graph){
#     plot.network()
#   }
#   
#   # Define a variable to keep track of the number of
#   # neuron updates, changes, and consecutive failed
#   # updates that occur
#   num.updates <- 0
#   num.changes <- 0
#   consec.failed.updates <- 0
#   
#   # define the while loop condition
#   condition <- TRUE
#   
#   # update the network n times
#   while (condition) {
#     
#     # increment num.updates
#     num.updates <- num.updates +1
#     
#     # select random neuron
#     num <-sample(1:N, 1)
#     
#     # find the new state and the updated state
#     act <- act.vector[num]
#     old.state <- network[num]
#     new.state <- act.to.state(act)
#     
#     # if the activation is nonzero and the states are different
#     # change the neuron to its new state, update the activation
#     # vector, increment num.changes, and set consec.failed,updates
#     # to zero
#     if(new.state != old.state){
#       
#       network[num] <<- new.state
#       act.vector <<- as.vector(weights %*% network)
#       
#       if(graph && (num.changes+1)%%graph.freq == 0){
#         plot.network()
#       }
#       
#       num.changes <- num.changes + 1
#       consec.failed.updates <- 0
#     }
#     
#     # otherwise count it as a failed update
#     else{
#       consec.failed.updates <- consec.failed.updates + 1
#     }
#     
#     # there's about a 90% cahnce that some neuron has not
#     # been updated is after 5N updates. This number changes very slightly with N.
#     # the derivation is at https://www.desmos.com/calculator/dyomwdesru.
#     if(consec.failed.updates > 5*N && is.stable(stab.threshold)){
#       condition <- FALSE
#       
#       num.updates <- num.updates - consec.failed.updates
#     }
#   }
#   
#   if(graph){
#     plot.network()
#     
#     # return num.cahnges and num.updates required to stabalize the network
#     print(c(num.changes, num.updates))
#   }
# }

