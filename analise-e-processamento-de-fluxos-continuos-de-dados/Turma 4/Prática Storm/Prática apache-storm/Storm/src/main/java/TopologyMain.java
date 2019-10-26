import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.topology.TopologyBuilder;


public class TopologyMain {
    public static void main(String[] args) throws InterruptedException {

        //Build Topology
        TopologyBuilder builder = new TopologyBuilder();
        builder.setSpout("YahooFinanceSpout", new yahooFinanceSpout());
        builder.setBolt("YahooFinanceBolt", new yahooFinanceBolt())
                .shuffleGrouping("YahooFinanceSpout");

        //Configuration
        Config conf = new Config();
        conf.setDebug(true);
        //conf.put("fileToWrite", "/home/CIT/pkassio/Documents/Storm/output/outputyf/output.txt");

        LocalCluster cluster = new LocalCluster();
        try{
            cluster.submitTopology("FinanceApiTopology", conf, builder.createTopology());
            Thread.sleep(50000);
        }
        finally {
            cluster.shutdown();
        }
    }


}
