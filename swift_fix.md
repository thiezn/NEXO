Plan several fixes to the Moretimer Swift App.

Make sure to use the xcode mcp to look up design guidelines and other apple documentation, and to build and test the app.

- There seems to be a strange concentric overlay over the tab bar icons with the tint color, and can't select anything. The search button is visible but also not selectable. 
- When navigating to a book, it does not scroll to the last read position, and instead starts at the beginning.
- When reading a book in 'paged' mode, sometimes the text seems to overlap with each other. Continuous mode does seem to work fine.
- The Thread card view should have a separate image per category.

## Header

- The title for the top level views (home, library, threads) is not aligned with the button, but it's positioned below. I want it aligned with the toolbar buttons.
- When scrolling up in a top level view, the title and buttons should scroll up and disappear, and when scrolling down they should reappear. See developer design guidelines if there is anything about this.
- The Avatar icon on the top level screens is not filling the full circle, but showing some of the glass background around it. I would like it to fill the whole circle, but keep the grouping of the avatar and '...' button if possible.

## WebSocket NexoService

The WebSocket client is not able to connect to the server, giving the following error: . I have a running websocket server listening on ws://127.0.0.1:6969 for you to test. I know the server works as i can connect to it using tthe nexo_client rust tool.


nw_socket_handle_socket_event [C1:2] Socket SO_ERROR [61: Connection refused]
nw_endpoint_flow_failed_with_error [C1 127.0.0.1:6969 in_progress socket-flow (satisfied (Path is satisfied), viable, interface: lo0)] already failing, returning
Connection failed: Connection failed: Waiting: The operation couldn’t be completed. (Network.NWError error 61 - Connection refused)

## Books

- The Library view has a separate edit button. Move this into the ... menu to declutter the UI.
- The Book view chapter headings are displayed double. I only want to show the chapter heading once.
- The book card view in the home screen and library view should use the book cover image.

## Settings 

- Add a avatar image edit flow when you click on the avatar in settings. It should also then allow you to select a different image.
- Add pinch to zoom for the avatar image when selecting a new one, or editing the existing, so we can zoom in to where it looks best in a circle
- Add the name and email from Apple sign in to the settings view if provided. If apple allows users to change their settings, make sure to add that to the settings view as well.
