
// Allows executable to find class Controller.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Defines class Controller that defines basic functionality to process a
 * an HTTP message.
 * @version 0.0
 * @author Tom Lever
 */
abstract class Controller {
    
    private final static Logger logger =
        Logger.getLogger(Controller.class.getName());
    
    /**
     * Defines constructor Controller.
     */
    public Controller() {
        logger.log(new LogRecord(Level.INFO,
            "Controller constructor: Started."));
    }
    
    /**
     * Declares abstract method process, which must be implemented to process
     * IceCreamApplicationMessages into Products.
     * @param messageToProcess
     */
    abstract public void process(IceCreamApplicationMessage messageToProcess);
}