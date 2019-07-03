using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.DataSet
{
    public class DataRow
    {

        internal double[] Data;
        internal double[] Labels;
        internal int LabelSize;
        internal int FeatureSize;
        internal double[] Features;
        public DataRow(double[] data, int labelSize)
        {
            int fs = data.Length - labelSize;
            if(fs <= 0 || labelSize <= 0)
            {
                throw new ArgumentException("Parameter labelSize should be an integer greater than 0 , and less than data.Length.");
            }
            this.Data = data;
            this.LabelSize = labelSize;
            this.FeatureSize = fs;
            Labels= data.Skip(FeatureSize).ToArray();
            Features= data.Take(FeatureSize).ToArray();
        }

        public double[] GetLabels()
        {
            return Labels;
        }

        public double[] GetFeatures()
        {
            return Features;
        }


        public double[] GetData()
        {
            return Data.Clone() as double[];
        }



        public double Get(int index)
        {
            return Data[index];
        }



    }
}
