using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Basic_Neuron_Network
{
    public enum layerType
    {
        HiddenLayer,
        OutputLayer
    }

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

        public double output { get { return afn(sumInputs); } }

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
        public const double eta = 0.001;
        public int numNeurons;
        public Neuron[] neurons;

        public void forwardProp(Layer prevLayer)
        {
            for (int i = 0; i < numNeurons; i++)
            {
                for (int j = 0; j < neurons[i].numInputs; j++)
                    neurons[i].input[j] = prevLayer.neurons[j].output;
                neurons[i].sumSetChanged();
            }
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
                neurons[i].shift += eta * neurons[i].error;
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

        public OutputLayer(int numNeurons, int numInputs) : base(numNeurons, numInputs)
        {
        }

        public void backProp(double[] target)
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

    public class NeuronNetwork
    {
        public List<Layer> layers;
        int numOutputs;
        int numInputs;
        int numFirstLayerNeurons;

        public NeuronNetwork(int numInputs, int numFirstLayerNeurons)
        {
            this.numInputs = numInputs;
            this.numFirstLayerNeurons = numFirstLayerNeurons;
            layers = new List<Layer>();
            layers.Add(new HiddenLayer(numFirstLayerNeurons, numInputs));
        }

        public void addLayer(int numNeurons, layerType layerType)
        {
            if (layerType == layerType.HiddenLayer)
                layers.Add(new HiddenLayer(numNeurons, (layers.Last()).numNeurons)); 
            if (layerType == layerType.OutputLayer)
                layers.Add(new OutputLayer(numNeurons, (layers.Last()).numNeurons));
            numOutputs = numNeurons;
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
                    {
                        layers[0].neurons[i].input[j] = input[d][j];
                        layers[0].neurons[i].sumSetChanged();
                    }
                forwardProp();
                for (int i = 0; i < numOutputs; i++)
                {
                    singleTestAcc += 1 - Math.Abs(target[d][i] - layers[layers.Count - 1].neurons[i].output);
                }
                accuracy += singleTestAcc / numOutputs;
            }
            return accuracy / numData;
        }

        public void trainSingle(double[] input, double[] target)
        {
            for (int i = 0; i < numFirstLayerNeurons; i++)
            {
                for (int j = 0; j < numInputs; j++)
                    layers[0].neurons[i].input[j] = input[j];
                layers[0].neurons[i].sumSetChanged();
            }
            forwardProp();
            ((OutputLayer)layers[layers.Count - 1]).backProp(target);
            for (int i = layers.Count - 2; i >= 0; i--)
                layers[i].backProp(layers[i + 1]);
            for (int i = 0; i < layers.Count; i++)
                layers[i].changeWeights();
        }

        public void train(int numTrainCycles, int numData, double[][] input, double[][] target)
        {
            int iRnd;
            for (int d = 0; d < numTrainCycles; d++)
            {
                Random rnd = new Random();
                iRnd = rnd.Next(0, numData - 1);
                trainSingle(input[iRnd], target[iRnd]);
            }
        }

        public double[] getAnswer(double[] input)
        {
            for (int i = 0; i < numFirstLayerNeurons; i++)
                for (int j = 0; j < numInputs; j++)
                {
                    layers[0].neurons[i].input[j] = input[j];
                    layers[0].neurons[i].sumSetChanged();
                }
            forwardProp();

            double[] output = new double[numOutputs];
            for (int i = 0; i < numOutputs; i++)
            {
                output[i] = layers[layers.Count - 1].neurons[i].output;
            }
            return output;
        }
    }
}
