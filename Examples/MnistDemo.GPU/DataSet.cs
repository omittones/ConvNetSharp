using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.GPU.Single;

namespace MnistDemo.GPU
{
    public class Batch : Tuple<Volume<float>, Volume<float>, int[]>
    {
        public bool Final { get; set; }
        public Batch(Volume<float> x, Volume<float> y, int[] labels) : base(x, y, labels)
        {
        }
    }

    internal class DataSet
    {
        private int start;
        private readonly Random _random = new Random(RandomUtilities.Seed);
        private readonly List<MnistEntry> trainImages;

        public DataSet(List<MnistEntry> trainImages)
        {
            this.trainImages = trainImages;

            for (var i = this.trainImages.Count - 1; i >= 0; i--)
            {
                var j = this._random.Next(i);
                var temp = this.trainImages[j];
                this.trainImages[j] = this.trainImages[i];
                this.trainImages[i] = temp;
            }
        }

        public int Count => this.trainImages.Count;

        public Batch NextBatch(int batchSize, Batch old = null)
        {
            const int w = 28;
            const int h = 28;
            const int numClasses = 10;

            var dataShape = new Shape(w, h, 1, batchSize);
            var labelShape = new Shape(1, 1, numClasses, batchSize);
            var labels = new int[batchSize];

            Volume<float> dataVolume;
            Volume<float> labelVolume;
            if (old != null && old.Item1.Shape.Equals(dataShape))
            {
                dataVolume = old.Item1;
                labelVolume = old.Item2;
                labelVolume.Clear();
            }
            else
            {
                dataVolume = BuilderInstance.Volume.SameAs(dataShape);
                labelVolume = BuilderInstance.Volume.SameAs(labelShape);
            }

            bool final = false;
            for (var i = 0; i < batchSize; i++)
            {
                var entry = this.trainImages[this.start];

                labels[i] = entry.Label;

                var j = 0;
                for (var y = 0; y < h; y++)
                {
                    for (var x = 0; x < w; x++)
                    {
                        dataVolume.Set(x, y, 0, i, entry.Image[j++] / 255.0f);
                    }
                }
                                
                labelVolume.Set(0, 0, entry.Label, i, 1.0f);

                this.start++;
                if (this.start == this.trainImages.Count)
                {
                    this.start = 0;
                    final = true;
                }
            }            

            return new Batch(dataVolume, labelVolume, labels)
            {
                Final = final
            };
        }
    }
}