
/*
 * Allows executable to find class Main.
 */
package com.mycompany.serverutilities;

import java.io.IOException;
import org.apache.commons.lang.StringUtils;

/**
 * Defines Main class that encapsulates static method main,
 * the entry point of this application.
 * This application instantiates a server that listens for messages.
 * @version 0.0
 * @author Tom Lever
 */
public class Main {
    
    /**
     * Defines the main method of this application.
     * @param args
     */
    public static void main(String[] args) throws IOException {
        
        System.out.println("Started method main.");
        
        // Check that portNumber is in command-line arguments.
        if ((args.length != 1) || (!StringUtils.isNumeric(args[0]))) {
            System.err.println(
                    "Usage error. Usage: java EchoServer <portNumber>");
            System.exit(1);
        }
        
        // Parse portNumber from command-line arguments.
        int portNumber = Integer.parseInt(args[0]);
        System.out.printf("User-specified Port number: %d.\n", portNumber);
        
        Server server = new Server(portNumber);
        /* 
         * TODO: Handle IOException when thrown by server constructor.
         */
        
        server.listenAndRespond();
    }
}
