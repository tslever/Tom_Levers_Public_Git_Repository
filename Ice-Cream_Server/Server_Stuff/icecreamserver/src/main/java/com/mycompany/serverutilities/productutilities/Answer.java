
// Allows handle to find class Answer.
package com.mycompany.serverutilities.productutilities;

// Import classes.
import java.util.HashMap;

/**
 * Defines class Answer, an answer of which is used to construct a response
 * to the client.
 * @version 0.0
 * @author Tom Lever
 */
public class Answer {
    
    private final HashMap<String, String> hashMap;
    
    /**
     * Defines constructor Answer, which initializes this.hashMap.
     */
    public Answer() {
        this.hashMap = new HashMap<>();
    }
    
    /**
     * Defines method putInHashMapAt, which puts value in this.hashMap at key.
     * @param key
     * @param value 
     */
    public void putInHashMapAt(String key, String value) {
        this.hashMap.put(key, value);
    }
    
    /**
     * Defines method getHashMap, which returns this.hashMap.
     * @return this.hashMap
     */
    public HashMap<String, String> getHashMap() {
        return this.hashMap;
    }
}
