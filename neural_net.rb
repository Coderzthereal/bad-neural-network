# My implementation of a neural net.
# Since I only know the basics of how neural nets work, this will probably be slow and inefficient.

# A network has a set of input nodes, a hidden layer, and a set of output nodes. It works in waves and is auto-propagating.
class Network
	attr_reader :inputs, :hidden_layers, :outputs, :records
	def initialize(inputs, hidden_layers, outputs, weights: nil) # inputs: list of input ids; hidden layers: [x, y] of size of hidden layers; outputs: list of output ids
		
		# EXPLANATION OF WEIGHTS
		<<-end
		
		EXPLANATION OF WEIGHTS
		
		Weights is a 2D array of weights to be assigned to the neurons in the network. It looks something like this:
		
		[
		[0.3, 0.57, 0, 1],
		[0.4, 0.281, 0.3, 1],
		[0.35, 0.9, 0.132]
		]
		
		This describes a neural network with 4 input neurons, one hidden layer with 4 neurons,
		and an output layer with 3 neurons.
		
		You can also describe the 2D array and have the network build it for you. Right now, the only supported shortcuts are:
		- 1 (the object)
			Fills a 2D array with 1s.
		- nil (the object)
			Fills a 2D array with random values between 0 and 1.
		
		end
		
		@inputs, @hidden_layers, @outputs, @records = [], [], [], []
		
		case weights
		when 1 # Fills weightmap with 1s, creating a neural network that doesn't change input values
			weights_for_init = [1]*(hidden_layers[0]+2) # Hidden layers x plus 2 for the input and output layers
			weights_for_init = [weights_for_init] * hidden_layers[1]
			
		when nil # To fill with random values, use nil
			weights_for_init = [nil]*(hidden_layers[0]+2) # Hidden layers x plus 2 for the input and output layers
			weights_for_init = [weights_for_init] * hidden_layers[1]
			weights_for_init.each do |x_layer| # replaces every nil with a random decimal between 0 and 1.
				x_layer.map! {|item| rand }
			end
			
		else
			# Assume that weights is a preprepared 2D array of weights for the neurons
			# For now, no input checking
			
			weights_for_init = weights
		end
		
		# Create neurons with weights
		# I really hope this works!
		
		inputs.size.times do |input_y|
			@inputs << InputNeuron.new(weights_for_init.transpose[0][input_y-1], inputs[input_y]) # had to use #transpose b/c the original weights are interpreted the same as your average 5th grade coordinate plane
		
		end
		
		hidden_layers[1].times do |layer|
			@hidden_layers << []
			hidden_layers[0].times do |neuron|
				@hidden_layers[layer] << Neuron.new(weights_for_init[layer][neuron-1])
			end
		end
		
		outputs.size.times do |input_y|
			@outputs << OutputNeuron.new(weights_for_init.transpose[-1][input_y-1], outputs[input_y])
		end
		
		# Connect newly-created neurons together to actually form the network
		# This will probably work the first time.
		# Hidden layer networking is done here because it's trivial to implement
		# either way.
		
		@inputs.map { |input_neuron| @hidden_layers[0].each { |hidden_neuron| input_neuron.link_to hidden_neuron } }
		
		@hidden_layers.transpose.each_with_index do |vertical_layer, x| # uses #transpose due to the orientation of the hidden layers.
			if x == (@hidden_layers.transpose.size - 1) # If this is the last layer of the hidden layers
				# link to the output neurons
				vertical_layer.each do |hidden_neuron|
					@outputs.each do |output_neuron|
						hidden_neuron.link_to output_neuron
					end
				end
			else
				# Otherwise link to the layer after this one
				vertical_layer.each do |hidden_neuron|
					@hidden_layers.transpose[x+1].each do |next_hidden_neuron|
						hidden_neuron.link_to next_hidden_neuron
					end
				end
			end
		end
		# Output neurons already taken care of in neuron initialization.
	end
	
	def wave(values) # values is a hash used to match the input values to the correct input neurons. For instance, to input a value of 0.5 to the 'height' neuron, you would have an entry for 'height' => 0.5.

		values.each do |id, value|
			@inputs.select{|a|a.id==id}[0].feed(value)
		end
		temp_records = {}
		@outputs.each do |output_neuron|
			output_neuron.next_wave.each {|k,v|temp_records[k]=v}
		end
		@records << temp_records
		temp_records
	end
end

class Neuron
	attr_accessor :weight
	attr_reader :triggers, :triggered_with_values # An array of values that this neuron was triggered with
	def initialize(weight)
		@weight = weight
		@triggers, @triggered_with_values = [], []
	end
	def link_to neuron
		@triggers << neuron
	end
	def trigger value # This method creates auto-propagation, where when a neuron is triggered it automatically triggers all of the neurons coming after it. This also means that for each wave, you can just trigger the input neurons and they will handle the rest by themselves.
		@triggered_with_values << value
		@triggers.each do |trigger|
			trigger.trigger value*weight
		end
	end
	def next_wave
		# resets @triggered_with_values, to prepare for the next wave; returns the average value from the last propagation
		# also prepares all of its connected neurons for the next wave, that way you can just reset the input neurons and the whole network will reset for the next wave
		var = average_value
		@triggers.each {|t| t.next_wave }
		@triggered_with_values = []
		return var
	end
	def average_value # Weighted
		(@triggered_with_values.sum / @triggered_with_values.size.to_f) * @weight
	end
	def add_trigger(neuron)
		@triggers << neuron
	end
	def inspect
		"#<#{self.class} w:#{@weight.to_s[0..6]}"
	end
end

class InputNeuron < Neuron
	attr_reader :id
	def initialize(weight, id)
		super(weight)
		@id = id
	end
	
	def feed(value) # Just semantics; I think it sounds better to "feed" the network than to "trigger" it
		trigger value
	end
end

class OutputNeuron < Neuron
	attr_accessor :id
	def initialize(weight, id)
		super(weight)
		@id = id
	end
	def trigger value
		@triggered_with_values << value
	end
	def next_wave
		{@id => super}
	end
end

__END__
#My first implementation, before I knew all of the basics of how neural nets work.
class TriggerObject
	def initialize(&proc)
		@proc = proc
	end
	def trigger(*args, &proc)
		@proc.call(*args, &proc)
	end
end

class NeuralNet
	attr_reader :width, :height
	attr_accessor :default_threshold, :default_weight, :default_proc, :default_input_proc, :default_output_proc
	class Neuron
		attr_reader :id
		def initialize(threshold, name, output_objs={}) # The output hash is a list of output triggers and the weight of the connection to them. Output format: [object/neuron, weight of connection]
			@output = output_objs
			@threshold = threshold
			@id = name
		end
		def trigger(value) # This method is how the neuron is triggered by the input and how it triggers the connected neurons as well.
			@output.each do |obj, weight|
				if value*weight > @threshold 
					obj.trigger(value*weight)
				end
			end
		end
	end
	class InputNeuron # Based on the original Neuron class, except it is used to feed information into the network.
		attr_reader :id
		def initialize(output_objs, name) # Notice how there is no threshold value for this type of neuron. For now, this is the only difference between this kind of neuron and a "normal" neuron.
			@output = output_objs
			@id = name
		end
		def trigger(value)
			@output.each do |obj, weight|
				obj.trigger(value*weight)
			end
		end
	end
	class OutputNeuron # Also based on the original Neuron class, but it is only connected to one output, which is a proc.
		attr_reader :id
		def initialize(threshold, output_proc, weight, name) # This kind of neuron does have a threshold, but it has an output proc that does something instead of a hash of output neurons. Because of that, there is only one weight.
			@threshold = threshold
			@output = output_proc
			@weight = weight
			@id = name
		end
		def trigger(value)
			if value*@weight < @threshold
				@output.call(value*@weight)
			end
		end
	end
	def initialize(size=[1, 5], default_threshold = 1, default_weight=1, &default_proc)
		@default_threshold = default_threshold
		@default_weight = default_weight
		@default_proc = default_proc
		@default_proc ||= Proc.new { |value| puts "Default output proc got: #{value}" }
		@default_input_proc ||= @default_proc
		@default_output_proc ||= @default_proc
		@neurons = []
		@input_neurons = []
		@output_neurons = []
		@width = size[0]
		@height = size[1]
	end
	def add_input_neuron(name, outputs)
		@input_neurons << InputNeuron.new(outputs, name)
	end
	def add_neuron(name, threshold, outputs)
		@neurons << Neuron.new(threshold, name, outputs)
	end
	# It is suggested to define the output neurons first if you are defining them manually, as they are the outputs for all of the other neurons.
	def add_output_neuron(name, threshold, weight, &output)
		@output_neurons << OutputNeuron.new(threshold, output, weight, name)
	end
	def define_neurons_with_defaults(default_output_proc=@default_proc, default_weight=@default_weight, default_threshold=@default_threshold) # Some explanation may be necessary--last_neurons_added is the actual list of last neurons added. last_neurons_added2 is the current column of neurons, and once they are all added, then it is assigned to last_neurons_added and cleared.
		# Defining output neurons
		@height.times do
			add_output_neuron('output_default', default_threshold, default_weight, &default_proc)
		end
		# Defining neurons
		last_neurons_added = []
		@height.times do
			last_neurons_added << add_neuron('neuron_default', default_threshold, @output_neurons)
		end
		@width.times do
			last_neurons_added2 = []
			(@height-1).times do
				last_neurons_added2 << add_neuron('neuron_default', default_threshold, last_neurons_added)
			end
			last_neurons_added = last_neurons_added2
			last_neurons_added2.clear
		end
		# Defining input neurons
		@height.times do
			add_input_neuron('input_default', last_neurons_added)
		end
	end
end