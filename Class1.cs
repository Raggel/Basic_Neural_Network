using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Basic_Neural_Network
{
    public class Neuron
    {
        protected double _sumInputs;
        protected bool _sumInputsChanged;

        public double sumInputs { get { if (_sumInputsChanged) return _calcSumInputs(); else return _sumInputs; } }

        protected double _calcSumInputs()
        {
            double sum = 0;
            for (int i = 0; i < numInputs; i++)
                sum += input[i] * weight[i];
            _sumInputs = sum + shift;
            _sumInputsChanged = false;
            return _sumInputs;
        }

        public void sumSetChanged()
        {
            _sumInputsChanged = true;
        }

        protected double _output;
        protected bool _outputChanged;

        public double output { get { if (_outputChanged) return _calcOutput(); else return _output; } }

        protected double _calcOutput()
        {
            _output = afn(sumInputs);
            _outputChanged = false;
            return _output;
        }

        public void outputSetChanged()
        {
            _outputChanged = true;
        }

        public int numInputs;
        public double[] input;

        public double afn(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public double pafn(double x)
        {
            double f = afn(x);
            return f * (1 - f);
        }

        public double error;
        public double shift;
        public double[] deltaW;
        public double[] weight;

        public Neuron(int numInputs)
        {
            this.numInputs = numInputs;
            weight = new double[numInputs];
            input = new double[numInputs];
            deltaW = new double[numInputs];
            Random rand = new Random();
            for (int i = 0; i < numInputs; i++)
                weight[i] = rand.NextDouble() - 0.5;
            shift = rand.NextDouble() - 0.5;
            _sumInputsChanged = true;
        }
    }

    public abstract class Layer
    {
        public const double eta = 1;
        public int numNeurons;
        public Neuron[] neurons;

        public void getInputs(Layer prevLayer)
        {
            for (int i = 0; i < numNeurons; i++)
                for (int j = 0; j < neurons[i].numInputs; j++)
                    neurons[i].input[j] = prevLayer.neurons[i].output;
        }

        public void forwardProp(Layer prevLayer)
        {
            getInputs(prevLayer);
        }

        public void changeWeights()
        {
            for (int i = 0; i < numNeurons; i++)
                for (int j = 0; j < neurons[i].numInputs; j++)
                    neurons[i].weight[j] += neurons[i].deltaW[j];
        }

        public  void backProp(Layer nextLayer)
        {
            for (int i = 0; i < numNeurons; i++)
            {
                double nextLayerError = 0;
                for (int j = 0; j < nextLayer.numNeurons; j++)
                    nextLayerError = nextLayerError + nextLayer.neurons[j].error * nextLayer.neurons[j].weight[i];
                neurons[i].error = nextLayerError * neurons[i].pafn(neurons[i].sumInputs);
                for (int j = 0; j < neurons[i].numInputs; j++)
                    neurons[i].deltaW[j] = eta * neurons[i].error * neurons[i].input[j];
                neurons[i].shift = neurons[i].shift + eta * neurons[i].error;
            }
        }
        public Layer(int numNeurons, int numInputs)
        {
            this.numNeurons = numNeurons;
            neurons = new Neuron[numNeurons];
            for (int i = 0; i < numNeurons; i++)
                neurons[i] = new Neuron(numInputs);
        }
    }

    public class HiddenLayer : Layer
    {
        public HiddenLayer(int numNeurons, int numInputs) : base(numNeurons, numInputs)
        {
        }
    }

    public class OutputLayer : Layer
    {
        public double[] target;

        public OutputLayer(int numNeurons, int numInputs) : base(numNeurons, numInputs)
        {
            target = new double[numNeurons];
        }

        public void backProp()
        {
            for (int i = 0; i < numNeurons; i++)
            {
                neurons[i].error = (target[i] - neurons[i].output) * neurons[i].pafn(neurons[i].sumInputs);
                for (int j = 0; j < neurons[i].numInputs; j++)
                    neurons[i].deltaW[j] = eta * neurons[i].error * neurons[i].input[j];
                neurons[i].shift += eta * neurons[i].error;
            }
        }
    }

    public class NeuralNetwork
    {
        List<Layer> layers;
        int numOutputs;
        int numInputs;
        int numFirstLayerNeurons;

        public NeuralNetwork(int numOutputs, int numInputs, int numFirstLayerNeurons, int numFirstLayerNeuronsInputs)
        {
            this.numOutputs = numOutputs;
            this.numInputs = numInputs;
            this.numFirstLayerNeurons = numFirstLayerNeurons;
            layers = new List<Layer>();
            layers.Add(new HiddenLayer(numFirstLayerNeurons, numFirstLayerNeuronsInputs));
        }

        public enum layerType
        {
            HiddenLayer,
            OutputLayer
        }

        public void addLayer(int numNeurons, layerType layerType)
        {
            if (layerType == layerType.HiddenLayer)
                layers.Add(new HiddenLayer(numNeurons, numOutputs));
            if (layerType == layerType.OutputLayer)
                layers.Add(new OutputLayer(numNeurons, numOutputs));
        }

        public void forwardProp()
        {
            for (int i = 1; i < layers.Count; i++)
                layers[i].forwardProp(layers[i-1]);
        }

        public double checkPerformance(int numData, double[][] input, double[][] target)
        {
            double accuracy = 0;
            for (int d = 0; d < numData; d++)
            {
                double singleTestAcc = 0;
                for (int i = 0; i < numFirstLayerNeurons; i++)
                    for (int j = 0; j < numInputs; j++)
                        layers[0].neurons[i].input[j] = input[d][j];
                forwardProp();
                for (int i = 0; i < numOutputs; i++)
                    singleTestAcc += 1 - Math.Abs(target[d][i] - layers[layers.Count - 1].neurons[i].output);
                accuracy += singleTestAcc / numOutputs;
            }
            return accuracy / numData;
        }

        public void train(int numData, double[][] input, double[][] target)
        {
            for (int d = 0; d < numData; d++)
            {
                for (int i = 0; i < numFirstLayerNeurons; i++)
                    for (int j = 0; j < numInputs; j++)
                        layers[0].neurons[i].input[j] = input[d][j];
                forwardProp();
                for (int i = 0; i < numOutputs; i++)
                    ((OutputLayer)layers[layers.Count - 1]).target[i] = target[d][i];
                ((OutputLayer)layers[layers.Count - 1]).backProp();
                for (int i = layers.Count - 2; i >= 0; i--)
                    layers[i].backProp(layers[i + 1]);
                for (int i = 0; i < layers.Count; i++)
                    layers[i].changeWeights();
            }
        }
    }
}
