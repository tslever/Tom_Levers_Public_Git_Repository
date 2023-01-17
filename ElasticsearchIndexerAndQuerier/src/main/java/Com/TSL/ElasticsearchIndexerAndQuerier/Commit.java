package Com.TSL.ElasticsearchIndexerAndQuerier;

import com.fasterxml.jackson.databind.JsonNode;

public class Commit {
    
    public String id;
    public String name;
    public JsonNode dateTimeRange;
    
    public Commit() {
        
    }
    
    public Commit(String skuToUse, String nameToUse, JsonNode dateTimeRangeToUse) {
        id = skuToUse;
        name = nameToUse;
        dateTimeRange = dateTimeRangeToUse;
    }
    
    public void setId(String skuToUse) {
        id = skuToUse;
    }
    
    public void setName(String nameToUse) {
        name = nameToUse;
    }
    
    public void setJsonNode(JsonNode dateTimeRangeToUse) {
        dateTimeRange = dateTimeRangeToUse;
    }
    
    public String getId() {
        return id;
    }
    
    public String getName() {
        return name;
    }
    
    public JsonNode getDateTimeRange() {
        return dateTimeRange;
    }
    
    public String toString() {
        return
            "{\n" +
            "    \"id\": \"" + id + "\",\n" +
            "    \"name\": \"" + name + "\",\n" +
            "    \"dateTimeRange\": \"" + dateTimeRange + "\"\n" +
            "}";
    }
}