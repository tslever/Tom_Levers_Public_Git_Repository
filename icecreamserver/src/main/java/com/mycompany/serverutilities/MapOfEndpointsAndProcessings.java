
// Allows executable to find class Main.
package com.mycompany.serverutilities;

// Import classes.
import java.util.HashMap;

/**
 * Defines class MapOfEndpointsAndProcessings that allows creation of a
 * mapOfEndpointsAndProcessings that contains a hashMap of specific
 * (endpoint, processing method for messages received by a client) key/value
 * pairs, and a method to get the hashMap.
 * @version 0.0
 * @author Tom Lever
 */
public class MapOfEndpointsAndProcessings {
    
    private final HashMap<String, InterfaceForProcessing> hashMap;
    
    public MapOfEndpointsAndProcessings() {
        this.hashMap = new HashMap<>();
        
        this.hashMap.put(
            "/one",
            (InterfaceForProcessing) () -> { System.out.println("/one"); }
        );
        
        this.hashMap.put(
            "/two",
            (InterfaceForProcessing) () -> { System.out.println("/two"); }
        );
    }
    
    public HashMap getHashMap() {
        return this.hashMap;
    }
}