using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicMultilayerPerceptron.Perceptron.ActivationFunction
{
    class Tanh : ActivationFunction
    {
        public override double GetDerivativeValue(double x)
        {
            double v= 1 - Math.Pow(GetValue(x), 2);
            if (Double.IsNaN(v))
            {
                v = 0;
            }
            return v;
        }


        public override double GetValue(double x)
        {
            double v= (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
            if (Double.IsNaN(v))
            {
                if (x < 0)
                {
                    v = -1;
                }
                else
                {
                    v = 1;
                }
            }
            return v;
        }
    }
}
