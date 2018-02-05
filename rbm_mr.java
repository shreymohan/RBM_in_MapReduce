import java.io.IOException;
import java.util.*;

import org.apache.hadoop.conf.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.*;

public class rbm_mr extends Configured implements Tool {
   public int N;
	public int n_visible;
	public int n_hidden;
	public double[][] W;
	public double[] hbias;
	public double[] vbias;
	public Random rng;
	static double error=0.0;
        static double addError=0.0;
        int epochCount=0;
        double lr = 0.1;
  public static void main(String args[]) throws Exception {
    int res = ToolRunner.run(new rbm_mr(), args);
    System.exit(res);
  }

  public int run(String[] args) throws Exception {
    Random rng = new Random(123);
                N=6
		
		//int training_epochs = 1000;
		//int k = 1;
		  
		N = 6;
		int test_N = 2;
		n_visible = 6;
	        n_hidden = 3;
        RBM rbm = new RBM(train_N, n_visible, n_hidden, null, null, null, rng);
    Path inputPath = new Path(args[0]);
    Path outputPath = null;
    while(n>1){
    epochCount++;
    outputPath=new Path(args[0]+1);
    Configuration conf = getConf();
    Job job = new Job(conf, this.getClass().toString());

    FileInputFormat.setInputPaths(job, inputPath);
    FileOutputFormat.setOutputPath(job, outputPath);

    job.setJobName("WordCount");
    job.setJarByClass(rbm_mr.class);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    job.setMapOutputKeyClass(DoubleArrayWritable.class);
    job.setMapOutputValueClass(DoubleArrayWritable1.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    
    job.setMapperClass(Map.class);
    job.setCombinerClass(Reduce.class);
    job.setReducerClass(Reduce.class);

    job.waitForCompletion(true);
    inputPath=outputPath;
  }
int[][] test_X = {
			{1, 1, 0, 0, 0, 0},
			{0, 0, 0, 1, 1, 0}
		};
		
		double[][] reconstructed_X = new double[test_N][n_visible];

		for(int i=0; i<test_N; i++) {
			rbm.reconstruct(test_X[i], reconstructed_X[i]);
			for(int j=0; j<n_visible; j++) {
				System.out.printf("%.5f ", reconstructed_X[i][j]);
			}
			System.out.println();
		}

return 1;
}
public static class Map extends Mapper<LongWritable, Text, CustomWritable, CustomWritable1> {
    //private final static IntWritable number = new IntWritable();
    //private final static IntWritable diff = new IntWritable();
     public init() {
		
	         rng = new Random(1234);
			
		
			W = new double[n_hidden][n_visible];
			double a = 1.0 / n_visible;
			
			for(int i=0; i<n_hidden; i++) {
				for(int j=0; j<n_visible; j++) {
					W[i][j] = uniform(-a, a); 
				}
				} 
		
		
			hbias = new double[n_hidden];
			for(int i=0; i<n_hidden; i++) hbias[i] = 0;
		
		
		
			vbias = new double[n_visible];
			for(int i=0; i<n_visible; i++) vbias[i] = 0;
			
                }
    
    @Override
    public void map(LongWritable key, IntWritable value,
                    Mapper.Context context) throws IOException, InterruptedException {
      double[] ph_mean = new double[n_hidden];
      int[] ph_sample = new int[n_hidden];
      double[] nv_means = new double[n_visible];
      int[] nv_samples = new int[n_visible];
      double[] nh_means = new double[n_hidden];
      int[] nh_samples = new int[n_hidden];
      int k=1;
      double[][] deltaW=new double[n_hidden][n_visible];
      double[] deltaH=new double[n_hidden];
      double[] deltaV=new double[n_visible];
     
       
      CustomWritable cw=new CustomWritable();
      CustomWritable1 cw1=new CustomWritable1();
      String str=value.toString();
      int[] input=new int[6];
      for(int i=0;i<str.length();i++){
       if(str.charAt(i)=='1')
       int[i]=1;
       else
       int[i]=0;
       }
       if(epochCount==1)
       init();
      sample_h_given_v(input, ph_mean, ph_sample);
      for(int step=0; step<k; step++) {
			if(step == 0) {
				gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
			} else {
				gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
			}
		}
     
           for(int i=0; i<n_hidden; i++) {
			for(int j=0; j<n_visible; j++) {
				// W[i][j] += lr * (ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
				 deltaW[i][j]= lr * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]) / N;
                                error+=Math.pow((input[j]-nv_means[j]),2);
			}
                         
                         
			}
		for(int i=0; i<n_hidden; i++) {
			deltaH[i] = lr * (ph_sample[i] - nh_means[i]) / N;
		}
                
                
		for(int i=0; i<n_visible; i++) {
			deltaV[i] = lr * (input[i] - nv_samples[i]) / N;
		}
                cw1.setW(deltaW);
                cw1.sethbias(deltaH);
                cw1.setvbias(deltaV);
                cw1.setnh(nh_means);
                cw1.setph(ph_mean);

                cw.setW(W);
                cw.sethbias(hbias);
                cw.setvbias(vbias);
                
    //Text key=new Text(W.toString()+"\t"+hbias.toString()+"\t"+vbias.toString());
    //Text value=new Text(deltaW.toString()+"\t"+deltaH.toString()+"\t"+deltaV.toString());
     context.write(cw,cw1);

	}
      
    
 public void gibbs_hvh(int[] h0_sample, double[] nv_means, int[] nv_samples, double[] nh_means, int[] nh_samples) {
	  sample_v_given_h(h0_sample, nv_means, nv_samples);
	  sample_h_given_v(nv_samples, nh_means, nh_samples);
	}
 public void sample_h_given_v(int[] v0_sample, double[] mean, int[] sample) {
		for(int i=0; i<n_hidden; i++) {
			mean[i] = propup(v0_sample, W[i], hbias[i]);
			sample[i] = binomial(1, mean[i]);
		}
	}

	public void sample_v_given_h(int[] h0_sample, double[] mean, int[] sample) {
		for(int i=0; i<n_visible; i++) {
			mean[i] = propdown(h0_sample, i, vbias[i]);
			sample[i] = binomial(1, mean[i]);
		}
	}
  public double propup(int[] v, double[] w, double b) {
		double pre_sigmoid_activation = 0.0;
		for(int j=0; j<n_visible; j++) {
			pre_sigmoid_activation += w[j] * v[j];
		}
		pre_sigmoid_activation += b;
		return sigmoid(pre_sigmoid_activation);
	}
	
	public double propdown(int[] h, int i, double b) {
	  double pre_sigmoid_activation = 0.0;
	  for(int j=0; j<n_hidden; j++) {
	    pre_sigmoid_activation += W[j][i] * h[j];
	  }
	  pre_sigmoid_activation += b;
	  return sigmoid(pre_sigmoid_activation);
	}

public double uniform(double min, double max) {
		return rng.nextDouble() * (max - min) + min;
	}
	
	public int binomial(int n, double p) {
		if(p < 0 || p > 1) return 0;
		
		int c = 0;
		double r;
		
		for(int i=0; i<n; i++) {
			r = rng.nextDouble();
			if (r < p) c++;
		}
		
		return c;
	}
	
	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}
  }
public static class CustomWritable1 extends Writable {

private double[][]W;

private double[] hbias;

private double[] vbias;

private double[] nh;

private double[] ph;
 
         public CustomWritable1(){
      setW(double[0][0]);
      sethbias(double[0][0]);
      setVbias(double[0][0]);
      setnh(double[0][0]);
      setph(double[0][0]);   
}

    public CustomWritable1(double[][]W, double[] hbias, double[] vbias, double[]nh,double[]ph){
this.W=w;
this.hbias=hbias;
this.vbias=vbias;
this.nh=nh;
this.ph=ph;
}
public void setW(double[][]W){
this.W=W;
}

public void sethbias(double[]hbias){
this.hbias=hbias;
}

public void setvbias(double[]vbias){
this.vbias=vbias;
} 
public void setnh(double[]nh){
this.nh=nh;
} 
public void setph(double[]ph){
this.ph=ph;
} 
public double[][] getW() {
        return W;
    }
public double[] gethbias() {
        return hbias;
    }
public double[] getvbias() {
        return vbias;
    }
public double[] getnh() {
        return nh;
    }
public double[] getph() {
        return ph;
    }
public void write(DataOutput out) throws IOException {
        out.writeInt(W.length);
        out.writeInt(W[0].length);
        
        for(int i=0;i<W.length;i++){
        for(int j=0;j<W[0].length;j++)
        out.writeDouble(W[i][j]);

}
}
        for(int i=0;i<W.length;i++)
        out.writeDouble(hbias[i]);
         for(int i=0;i<W[0].length;i++)
        out.writeDouble(vbias[i]);
        for(int i=0;i<W.length;i++)
        out.writeDouble(nh[i])
        for(int i=0;i<W.length;i++)
        out.writeDouble(ph[i])        
}

    public void readFields(DataInput in) throws IOException {
        int len=in.readInt();
        int breadth=in.readInt();
        W=new double[len][breadth];
        hbias=new double[len];
        vbias=new double[breadth];
        nh=new double[len];
        ph=new double[len];
        for(int i=0;i<len;i++){
        for(int j=0;j<breadth;j++){
         W[i][j]=in.readDouble();
         }
         }
        for(int i=0;i<len;i++)
        hbias[i]=in.readDouble();
        for(int i=0;i<breadth;i++)
        vbias[i]=in.readDouble();
        for(int i=0;i<len;i++)
        nh[i]=in.readDouble();
        for(int i=0;i<len;i++)
        ph[i]=in.readDouble();        
}
}
public static class CustomWritable implements WritableComparable<CustomWritable> {

private double[][]W;

private double[] hbias;

private double[] vbias;
 
         public CustomWritable(){
      setW(double[0][0]);
      sethbias(double[0][0]);
      setVbias(double[0][0]);   
}

    public CustomWritable(double[][]W, double[] hbias, double[] vbias){
this.W=w;
this.hbias=hbias;
this.vbias=vbias;
}
public void setW(double[][]W){
this.W=W;
}

public void sethbias(double[]hbias){
this.hbias=hbias;
}

public void setvbias(double[]vbias){
this.vbias=vbias;
} 
public double[][] getW() {
        return W;
    }
public double[] gethbias() {
        return hbias;
    }
public double[] getvbias() {
        return vbias;
    }
public void write(DataOutput out) throws IOException {
        out.writeInt(W.length);
        out.writeInt(W[0].length);
        
        for(int i=0;i<W.length;i++){
        for(int j=0;j<W[0].length;j++)
        out.writeDouble(W[i][j]);

}
}
        for(int i=0;i<W.length;i++)
        out.writeDouble(hbias[i]);
         for(int i=0;i<W[0].length;i++)
        out.writeDouble(vbias[i]);
        }

    public void readFields(DataInput in) throws IOException {
        int len=in.readInt();
        int breadth=in.readInt();
        W=new double[len][breadth];
        hbias=new double[len];
        vbias=new double[breadth];
        for(int i=0;i<len;i++){
        for(int j=0;j<breadth;j++){
         W[i][j]=in.readDouble();
         }
         }
        for(int i=0;i<len;i++)
        hbias[i]=in.readDouble();
        for(int i=0;i<breadth;i++)
        vbias[i]=in.readDouble();
        }

    public int compareTo(CustomWritable o) {
    int cmp=0;
for(int i=0;i<W.length;i++){
for(int j=0;j<W[0].length;j++){   
cmp+=W[i][j].compareTo(o.W[i][j])
    }
}
return cmp;
}

}
public static class CustomWritable2 implements WritableComparable<CustomWritable2> {

private double[][]W;

private double[] hbias;

private double[] vbias;

 
         public CustomWritable2(){
      setW(double[0][0]);
      sethbias(double[0][0]);
      setVbias(double[0][0]);
         
}

    public CustomWritable2(double[][]W, double[] hbias, double[] vbias){
this.W=w;
this.hbias=hbias;
this.vbias=vbias;
}
public void setW(double[][]W){
this.W=W;
}

public void sethbias(double[]hbias){
this.hbias=hbias;
}

public void setvbias(double[]vbias){
this.vbias=vbias;
} 

public double[][] getW() {
        return W;
    }
public double[] gethbias() {
        return hbias;
    }
public double[] getvbias() {
        return vbias;
    }

public void write(DataOutput out) throws IOException {
        out.writeInt(W.length);
        out.writeInt(W[0].length);
        
        for(int i=0;i<W.length;i++){
        for(int j=0;j<W[0].length;j++)
        out.writeDouble(W[i][j]);

}
}
        for(int i=0;i<W.length;i++)
        out.writeDouble(hbias[i]);
         for(int i=0;i<W[0].length;i++)
        out.writeDouble(vbias[i]);
               
}

    public void readFields(DataInput in) throws IOException {
        int len=in.readInt();
        int breadth=in.readInt();
        W=new double[len][breadth];
        hbias=new double[len];
        vbias=new double[breadth];
       
        for(int i=0;i<len;i++){
        for(int j=0;j<breadth;j++){
         W[i][j]=in.readDouble();
         }
         }
        for(int i=0;i<len;i++)
        hbias[i]=in.readDouble();
        for(int i=0;i<breadth;i++)
        vbias[i]=in.readDouble();
             
}
 public int compareTo(CustomWritable o) {
    int cmp=0;
for(int i=0;i<W.length;i++){
for(int j=0;j<W[0].length;j++){   
cmp+=W[i][j].compareTo(o.W[i][j])
    }
}
return cmp;
}
public String toString(){
String str="";
for(int i=0;i<len;i++){
str=Arrays.toString(W[i]);
}
str+=hbias;
str+=vbias;
return str;
}
}
public static class Reduce extends Reducer<CustomWritable,CustomWritable1, Text, IntWritable> {

    @Override
    public void reduce(CustomWritable key, Iterable<CustomWritable1> values, Context context) throws IOException, InterruptedException {
      double[][]W=new double[n_hidden][n_visible];
      double[] hbias=new double[n_hidden];
      double[] vbias=new double[n_visible]; 
      double[][] deltaW=new double[n_hidden][n_visible];
      double[] deltaH=new double[n_hidden];
      double[] deltaV=new double[n_visible];
      double[] ph_means=new double[n_hidden];
      double[] nh_means=new double[n_hidden];
     CustomWritable2 cw2=new ,CustomWritable2();
    W=key.getW();
    hbias=key.gethbias();
    vbias=key.getvbias();   
   for (CustomWritable1 val: values) {
 deltaW=val.getW();
 deltaH=val.gethbias();
 deltaV=val.getvbias();
 ph_means=val.getph();
 nh_means=val.getnh();
for(int i=0; i<n_hidden; i++) {
for(int j=0; j<n_visible; j++) {
W+=deltaW;
hbias+=deltaH;

}
error+=Math.pow((ph_means[i]-nh_means[i]),2);
}
for(int i=0; i<n_visible; i++)
vbias+=deltaV;
}
cw2.setW(W);
cw2.sethbias(hbias);
cw2.setvbias(vbias);

      context.write(key, new IntWritable(sum));
    }
  }
}

