using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.DataSet
{
    public class DataSet
    {

        internal List<DataRow> DataRows;
        public int FeatureSize { get;private set; }
        public int LabelSize { get; private set; }
        public DataSet(String path , char separator , int labelSize , bool hasHeaders)
        {
            var query = Enumerable.Empty<Double[]>(); ;
            LabelSize = labelSize;
            if (hasHeaders)
            {
                query = File.ReadLines(path).Skip(1).Select(i => Array.ConvertAll<String, Double>(i.Split(separator), Double.Parse));
            }
            else
            {

            query = File.ReadLines(path).Select(i => Array.ConvertAll<String, Double>(i.Split(separator), Double.Parse));
            }
            int size = query.First().Length;
            FeatureSize = size - LabelSize;
            if(query.Any(i=> i.Length != size))
            {
                throw new ArgumentException("All lines in the provided file should have the same number of values.");
            }
            DataRows = new List<DataRow>();
            foreach(double[] data in query)
            {
                DataRows.Add(new DataRow(data, labelSize));
            }
        }




        public List<List<DataRow>> Batch(int batchSize)
        {
            List<List<DataRow>> groups = new List<List<DataRow>>();
            
            List<DataRow> temp = new List<DataRow>();
            for (int i = 0; i < DataRows.Count; i++)
            {
                temp.Add(DataRows[i]);
                if (i == DataRows.Count - 1)
                {
                    //Console.WriteLine(temp.Count);
                    groups.Add(temp);
                }
                else
                {
                    if (temp.Count == batchSize)
                    {
                        //Console.WriteLine(temp.Count);
                        groups.Add(temp);
                        temp = new List<DataRow>();
                    }
                }
            }
            
            return groups;
        }



        public List<DataRow> NextBatch(int batchSize)
        {
           //Console.WriteLine(DataRows.Count);
           
            if (batchSize >= DataRows.Count)
            {
                return DataRows;
            }
            else
            {

                
                List<DataRow> all = DataRows.ToList();
                List<DataRow> ret = new List<DataRow>();
                Random rand = new Random();
                while (ret.Count != batchSize)
                {
                    //Console.WriteLine(ret.Count);
                    var num = rand.Next(0, all.Count);
                    ret.Add(DataRows[num]);
                    all.RemoveAt(num);
                }
                return ret;
            }
            
        }


        public void Shuffle()
        {
            Random rng = new Random();
            int n = DataRows.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                DataRow value = DataRows[k];
                DataRows[k] = DataRows[n];
                DataRows[n] = value;
            }
        }


    }
}
