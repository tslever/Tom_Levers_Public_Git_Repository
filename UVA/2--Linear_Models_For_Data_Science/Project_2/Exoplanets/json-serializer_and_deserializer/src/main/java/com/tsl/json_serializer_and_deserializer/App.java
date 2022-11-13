package com.tsl.json_serializer_and_deserializer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.Set;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws IOException
    {
    	InputStream inputStream = App.class.getResourceAsStream("Terrestrial_Exoplanets.json");
    	InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
    	BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
    	StringBuilder stringBuilder = new StringBuilder();
    	String line;
    	while ((line = bufferedReader.readLine()) != null) {
    		stringBuilder.append(line).append("\n");
    	}
    	try (PrintWriter printWriter = new PrintWriter("Terrestrial_Exoplanets.csv")) {
	    	String jsonObjectWithArrayOfTerrestrialExoplanetsAsString = stringBuilder.toString();
	        JsonObject jsonObjectWithArrayOfTerrestrialExoplanets = JsonParser.parseString(jsonObjectWithArrayOfTerrestrialExoplanetsAsString).getAsJsonObject();
	        JsonArray jsonArrayOfTerrestrialExoplanets = jsonObjectWithArrayOfTerrestrialExoplanets.get("items").getAsJsonArray();
	        stringBuilder = new StringBuilder();
	        JsonObject terrestrialExoplanet = jsonArrayOfTerrestrialExoplanets.get(0).getAsJsonObject();
	        Iterator<String> iterator = terrestrialExoplanet.keySet().iterator();
	        while (iterator.hasNext()) {
	        	String key = iterator.next();
	        	stringBuilder.append(key);
	        	if (iterator.hasNext()) {
	        		stringBuilder.append(",");
	        	}
	        }
	        String header = stringBuilder.toString();
	        printWriter.println(header);
	        for (int i = 0; i < jsonArrayOfTerrestrialExoplanets.size(); i++) {
	        	terrestrialExoplanet = jsonArrayOfTerrestrialExoplanets.get(i).getAsJsonObject();
	        	stringBuilder = new StringBuilder();
	        	iterator = terrestrialExoplanet.keySet().iterator();
	        	while (iterator.hasNext()) {
	        		String key = iterator.next();
	        		JsonElement value = terrestrialExoplanet.get(key);
	        		if (!(value instanceof JsonNull)) {
	        			if (value instanceof JsonArray) {
	        			    String valueAsString = (new Gson()).toJson(value);
	        			    stringBuilder.append(valueAsString.replaceAll(",", "COMMA").replaceAll("\\r", "CARRIAGE_RETURN").replaceAll("\\n", "NEWLINE"));
	        			} else {
	        			    stringBuilder.append(value.getAsString().replaceAll(",", "COMMA").replaceAll("\\r", "CARRIAGE_RETURN").replaceAll("\\n", "NEWLINE"));
	        			}
	        		} else {
	        			stringBuilder.append("null");
	        		}
	        		if (iterator.hasNext()) {
	        			stringBuilder.append(",");
	        		}
	        	}
	        	String record = stringBuilder.toString();
	        	printWriter.println(record);
	        }
    	}
    }
}
