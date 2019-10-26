import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.generated.StormTopology;
import org.apache.storm.topology.TopologyBuilder;


public class Topology3Bolts {
    public static void main(String[] args) throws InterruptedException {

        //Build Topology
        TopologyBuilder builder = new TopologyBuilder();
        builder.setSpout("YahooFinanceSpout", new yahooFinanceSpout());
        builder.setBolt("YahooFinanceBolt", new yahooFinanceBolt(),3)
                .shuffleGrouping("YahooFinanceSpout");

        //Configuration
        Config conf = new Config();
        conf.setDebug(true);
        conf.put("fileToWrite", "/home/CIT/pkassio/Documents/Storm/output/output.txt");

        //Submit Topology to cluster
        LocalCluster cluster = new LocalCluster();
        try{
            cluster.submitTopology("FinanceApiTopology", conf, builder.createTopology());
            Thread.sleep(5000);
        }
        finally {
            cluster.shutdown();
        }
    }
}
