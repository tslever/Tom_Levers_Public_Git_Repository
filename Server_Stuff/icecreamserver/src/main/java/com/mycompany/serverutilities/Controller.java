
// Allows executable to find class Controller.
package com.mycompany.serverutilities;

// Imports classes.
import java.time.LocalDateTime;

/**
 * Defines class Controller that encapsulates functionality to process a
 * an HTTP message.
 * @version 0.0
 * @author Tom Lever
 */
public class Controller {
    
    public Controller() {
        
    }
    
    /**
     * Defines method process, which reacts to a method calling it.
     */
    public void process() {
        System.out.println("Controller.process: Started.");
        
        System.out.printf(
            "Controller.process: Hi, at %s!\n", LocalDateTime.now());
    }
}