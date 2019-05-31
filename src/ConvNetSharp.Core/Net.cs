using System;
using System.Collections.Generic;
using System.IO;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core
{
    public class Net<T> : INet<T> where T : struct, IEquatable<T>, IFormattable
    {
        private List<LayerBase<T>> layers = new List<LayerBase<T>>();
        public IReadOnlyList<LayerBase<T>> Layers => layers;

        public INet<T> Clone()
        {
            var clone = new Net<T>();
            foreach (var layer in this.Layers)
                clone.AddLayer(layer.Clone());
            CopyParameters(clone);
            return clone;
        }

        public void CopyParameters(INet<T> to)
        {
            var source = this.GetParametersAndGradients();
            var destination = to.GetParametersAndGradients();
            for (var i = 0; i < source.Count; i++)
            {
                CopyVolume(source[i].Volume, destination[i].Volume);
                CopyVolume(source[i].Gradient, destination[i].Gradient);
                destination[i].L1DecayMul = source[i].L1DecayMul;
                destination[i].L2DecayMul = source[i].L2DecayMul;
            }
        }

        private void CopyVolume(Volume<T> from, Volume<T> to)
        {
            to.Clear();
            from.DoAdd(to, to);
        }

        public Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            var activation = this.layers[0].DoForward(input, isTraining);

            for (var i = 1; i < this.layers.Count; i++)
            {
                var layer = this.layers[i];
                activation = layer.DoForward(activation, isTraining);
            }

            return activation;
        }

        public T GetCostLoss(Volume<T> input, Volume<T> y)
        {
            Forward(input);

            var lastLayer = this.layers[this.layers.Count - 1] as ILastLayer<T>;
            if (lastLayer != null)
            {
                T loss;
                lastLayer.Backward(y, out loss);
                return loss;
            }

            throw new Exception("Last layer doesn't implement ILastLayer interface");
        }

        public T Backward(Volume<T> y)
        {
            var n = this.layers.Count;
            var lastLayer = this.layers[n - 1] as ILastLayer<T>;
            if (lastLayer != null)
            {
                T loss;
                lastLayer.Backward(y, out loss); // last layer assumed to be loss layer
                for (var i = n - 2; i >= 0; i--)
                {
                    var lastInputGradients = this.layers[i + 1].InputActivationGradients;
                    var thisLayer = this.layers[i];
                    thisLayer.Backward(lastInputGradients);
                }

                return loss;
            }

            throw new Exception("Last layer doesn't implement ILastLayer interface");
        }

        public int[] GetPrediction()
        {
            // this is a convenience function for returning the argmax
            // prediction, assuming the last layer of the net is a softmax
            var softmaxLayer = this.layers[this.layers.Count - 1] as SoftmaxLayer<T>;
            if (softmaxLayer == null)
            {
                throw new Exception("GetPrediction function assumes softmax as last layer of the net!");
            }

            var activation = softmaxLayer.OutputActivation;
            var N = activation.Shape.Dimensions[3];
            var C = activation.Shape.Dimensions[2];
            var result = new int[N];

            for (var n = 0; n < N; n++)
            {
                var maxv = activation.Get(0, 0, 0, n);
                var maxi = 0;

                for (var i = 1; i < C; i++)
                {
                    var output = activation.Get(0, 0, i, n);
                    if (Ops<T>.GreaterThan(output, maxv))
                    {
                        maxv = output;
                        maxi = i;
                    }
                }

                result[n] = maxi;
            }

            return result;
        }

        public List<ParametersAndGradients<T>> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients<T>>();
            foreach (var layer in this.Layers)
            {
                var parameters = layer.GetParametersAndGradients();
                response.AddRange(parameters);
            }
            return response;
        }

        public void AddLayer(LayerBase<T> layer)
        {
            int inputWidth = 0, inputHeight = 0, inputDepth = 0;
            LayerBase<T> lastLayer = null;

            if (this.layers.Count > 0)
            {
                var last = layers.Count - 1;
                inputWidth = this.layers[last].OutputWidth;
                inputHeight = this.layers[last].OutputHeight;
                inputDepth = this.layers[last].OutputDepth;
                lastLayer = this.layers[last];

                layer.Init(inputWidth, inputHeight, inputDepth);
            }
            else if (!(layer is InputLayer<T>))
            {
                throw new ArgumentException("First layer should be an InputLayer");
            }

            var classificationLayer = layer as IClassificationLayer;
            if (classificationLayer != null)
            {
                var fullConnLayer = lastLayer as FullyConnLayer<T>;
                if (fullConnLayer == null)
                {
                    throw new ArgumentException(
                        $"Previously added layer should be a FullyConnLayer with {classificationLayer.ClassCount} Neurons");
                }

                if (fullConnLayer.NeuronCount != classificationLayer.ClassCount)
                {
                    throw new ArgumentException(
                        $"Previous FullyConnLayer should have {classificationLayer.ClassCount} Neurons");
                }
            }

            if (layer is ReluLayer<T> || layer is LeakyReluLayer<T>)
            {
                if (lastLayer is IDotProductLayer<T> dotProductLayer)
                {
                    // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.

                    //commented out to reproduce 0.2.0 bug
                    //dotProductLayer.BiasPref = (T)Convert.ChangeType(0.1, typeof(T)); // can we do better?
                }
            }
            
            this.layers.Add(layer);
        }

        public void Dump(string filename)
        {
            using (var stream = File.Create(filename))
            using (var sw = new StreamWriter(stream))
            {
                for (var index = 0; index < this.layers.Count; index++)
                {
                    var layerBase = this.layers[index];
                    sw.WriteLine($"=== Layer {index}");
                    sw.WriteLine("Input");
                    sw.Write(layerBase.InputActivation.ToString());

                    var conv = layerBase as ConvLayer<T>;
                    if (conv != null)
                    {
                        sw.WriteLine("Filter");
                        sw.Write(conv.Filters.ToString());

                        sw.WriteLine("Bias");
                        sw.Write(conv.Bias.ToString());
                    }

                    var full = layerBase as FullyConnLayer<T>;
                    if (full != null)
                    {
                        sw.WriteLine("Filter");
                        sw.Write(full.Filters.ToString());

                        sw.WriteLine("Bias");
                        sw.Write(full.Bias.ToString());
                    }
                }
            }
        }

        public Volume<T> Forward(Volume<T>[] inputs, bool isTraining = false)
        {
            return Forward(inputs[0], isTraining);
        }

        public static Net<T> FromData(IDictionary<string, object> dico)
        {
            var net = new Net<T>();

            var layers = dico["Layers"] as IEnumerable<IDictionary<string, object>>;
            foreach (var layerData in layers)
            {
                var layer = LayerBase<T>.FromData(layerData);
                net.layers.Add(layer);
            }

            return net;
        }

        public Dictionary<string, object> GetData()
        {
            var dico = new Dictionary<string, object>();
            var layers = new List<Dictionary<string, object>>();

            foreach (var layer in this.Layers)
            {
                layers.Add(layer.GetData());
            }

            dico["Layers"] = layers;

            return dico;
        }
    }
}