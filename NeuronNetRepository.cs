using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Basic_Neuron_Network
{
    public class NeuronNetRepository
    {
        string fileName;

        public NeuronNetRepository(string fileName)
        {
            this.fileName = fileName;
        }

        private void saveNeuron(Neuron neuron, BinaryWriter bw)
        {
            foreach (double w in neuron.weight)
                bw.Write(w);
            bw.Write(neuron.shift);
        }

        private void loadNeuron(Neuron neuron, BinaryReader br)
        {
            for (int i = 0; i < neuron.numInputs; i++)
                neuron.weight[i] = br.ReadDouble();
            neuron.shift = br.ReadDouble();
        }

        private void saveLayer(Layer layer, BinaryWriter bw)
        {
            foreach (Neuron n in layer.neurons)
                saveNeuron(n, bw);
        }

        private void loadLayer(Layer layer, BinaryReader br)
        {
            for (int i = 0; i < layer.numNeurons; i++)
                loadNeuron(layer.neurons[i], br);
        }

        public void saveNet(NeuronNetwork net)
        {
            BinaryWriter bw = new BinaryWriter(File.OpenWrite(fileName));
            foreach (Layer l in net.layers)
                saveLayer(l, bw);
            bw.Write(net.trainCount);
            bw.Write(net.eta);
            bw.Close();
            bw.Dispose();
        }

        public void loadNet(NeuronNetwork net)
        {
            BinaryReader br = new BinaryReader(File.OpenRead(fileName));
            for (int i = 0; i < net.layers.Count; i++)
                loadLayer(net.layers[i], br);
            net.trainCount = br.ReadInt32();
            net.eta = br.ReadDouble();
            br.Close();
            br.Dispose();
        }
    }
}
