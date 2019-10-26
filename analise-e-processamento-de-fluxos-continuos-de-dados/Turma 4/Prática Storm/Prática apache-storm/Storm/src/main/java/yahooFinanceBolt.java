import org.apache.storm.task.TopologyContext;
import org.apache.storm.topology.BasicOutputCollector;
import org.apache.storm.topology.OutputFieldsDeclarer;
import org.apache.storm.topology.base.BaseBasicBolt;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Tuple;
import org.apache.storm.tuple.Values;

import java.io.PrintWriter;
import java.util.Map;

public class yahooFinanceBolt extends BaseBasicBolt {

    public void prepare(Map stormConf, TopologyContext context) {}


    public void execute(Tuple input,BasicOutputCollector collector) {
        String symbol = input.getValue(0).toString();
        String timestamp =  input.getString(1);

        Double price = (Double) input.getValueByField("price");
        Double prevClose = input.getDoubleByField("prev_close");

        Boolean gain = true;

        if (price<=prevClose) {
            gain = false;
        }
        System.out.println(symbol);
        System.out.println(timestamp);
        System.out.println(price);
        System.out.println(price);
        System.out.println(gain);

        collector.emit(new Values(symbol, timestamp, price,gain));
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("company", "timestamp", "price","gain"));
    }

    public void cleanup() {}
}