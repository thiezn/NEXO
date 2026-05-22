---
name: swift-ux
description: Use when creating new SwiftUI UX components in the Moretimer project. Covers component structure, theming, and integration with the app environment.
---

# Moretimer UX Component System

All UX code lives under `Moretimer/Moretimer/UX/`. The project uses `fileSystemSynchronizedGroups` so new files auto-appear in Xcode.

## Folder Structure

```
UX/
  Icons/AppIcon.swift           - Centralized SF Symbol constants
  Theme/
    ThemeColors.swift           - 7-color struct + EnvironmentKey (\.themeColors)
    AppearanceMode.swift        - Light/dark/auto enum
    AppTheme.swift              - 4 themes with light+dark color palettes
    ThemeManager.swift          - @Observable, persists to UserDefaults, ThemeResolvingModifier
  Components/
    AvatarView.swift            - Profile avatar with image/initials/icon fallbacks
    LargeCard.swift             - Portrait card (~160x240) with text overlay
    SmallCard.swift             - Square card (~120x120) with text below
    SectionHeader.swift         - Title + optional action button
    EmptyStateView.swift        - ContentUnavailableView wrapper
    CategoryBadge.swift         - Capsule badge
    MessageInputBar.swift       - Text field + send button
  Menus/AppMenu.swift           - 3-section structured menu
  Toolbars/ToolbarStyles.swift  - TopLevelToolbarContent, DetailToolbarContent
  Navigation/
    AppDestination.swift        - Universal navigation enum
    AppNavigationModifier.swift - Shared navigationDestination + zoom transitions
```

## Key Conventions

- Access theme colors via `@Environment(\.themeColors) private var themeColors` (not ThemeManager directly)
- Access namespace via `@Environment(\.appNamespace) private var namespace` (always optional `Namespace.ID?`)
- Use `AppIcon.constantName` for all SF Symbols — never hardcode symbol strings in views
- Navigation uses `AppDestination` enum with `NavigationLink(value: AppDestination.book(id))` — never raw `PersistentIdentifier`
- Use `.matchedTransitionSource(id:in:)` with optional namespace for zoom transitions (safe nil-handling built in)
- Liquid Glass: `.glassEffect(.regular.tint(color.opacity(0.15)), in: .rect(cornerRadius:))`
- All `@Observable` classes use `@MainActor`

## Creating New Components

Place new components in the appropriate subfolder. Follow this pattern:

```swift
import SwiftUI

struct MyComponent: View {
    // Required props first (let), optional props after (var with defaults)
    let title: String
    var subtitle: String?
    var tint: Color = .clear

    var body: some View {
        // Implementation
    }
}
```

When the component needs theme colors, use the environment:

```swift
@Environment(\.themeColors) private var themeColors
```

## Component Reference

### AppIcon

Centralized SF Symbol names. Add new icons here instead of hardcoding strings.

```swift
// Usage
Image(systemName: AppIcon.add)
Button("Edit", systemImage: AppIcon.edit) { }
```

Categories: Actions, Thread Actions, Book Actions, Navigation, Profile, Status, Themes, Appearance.

### Theme System

**ThemeColors** — 7 colors: `primary`, `secondary`, `accent`, `backgroundPrimary`/`Secondary`/`Tertiary`/`Quaternary`.

**AppTheme** — 4 themes (happyPink, hackerDark, gamerFlashy, romanticGreyRed), each with `lightColors` and `darkColors`. Uses `#if canImport(UIKit)` for cross-platform.

**AppearanceMode** — `auto`/`light`/`dark`, returns `ColorScheme?` via `.colorScheme`.

**ThemeManager** — Persists theme + appearance to UserDefaults. Use `.resolveThemeColors()` modifier on root view to inject `\.themeColors`.

```swift
// In MoretimerApp.swift (already applied)
.preferredColorScheme(themeManager.preferredColorScheme)
.resolveThemeColors()
```

### LargeCard

Portrait card with full-bleed image and gradient text overlay.

```swift
LargeCard(
    imageData: book.images.first?.imageData,  // Data?
    placeholderIcon: AppIcon.bookFilled,       // SF Symbol fallback
    subtext: book.author,                      // optional top line
    title: book.title,                         // required
    description: "80% complete",               // optional bottom line
    tint: themeColors.accent,                  // glass tint
    width: 160, height: 240                    // defaults
)
```

### SmallCard

Square card with text below the image.

```swift
SmallCard(
    imageData: nil,
    placeholderIcon: AppIcon.threads,
    title: thread.title,
    subtitle: thread.category,    // optional
    tint: themeColors.accent,
    size: 120                     // default
)
```

### AvatarView

Adaptive avatar: image > initials (size>32) > system icon (size<=32).

```swift
AvatarView(imageData: profile.avatarImageData, initials: "MM", size: 28)
```

### SectionHeader

Title + optional trailing action.

```swift
SectionHeader("Continue Reading", actionLabel: "See All") {
    navManager.selectedTab = .books
}
```

### EmptyStateView

Wraps `ContentUnavailableView` with optional action button.

```swift
EmptyStateView(
    "No Books Yet",
    systemImage: AppIcon.books,
    description: "Import an ePub to get started.",
    actionLabel: "Import",
    action: { showImporter = true }
)
```

### CategoryBadge

Capsule badge for tags/categories.

```swift
CategoryBadge(text: "Fiction", color: .blue)
```

### MessageInputBar

Multi-line text field + send button.

```swift
@State private var messageText = ""
MessageInputBar(text: $messageText, placeholder: "Message...") { content in
    sendMessage(content)
}
```

### AppMenu

3-section structured menu. All sections optional.

```swift
AppMenu(
    quickActions: [                                    // Horizontal icons (ControlGroup palette)
        MenuAction(title: "Share", icon: AppIcon.share) { share() }
    ],
    listSections: [                                    // Vertical items, sections separated by dividers
        [MenuAction(title: "Import ePub", icon: AppIcon.importBook) { importBook() }],
        [MenuAction(title: "Settings", icon: AppIcon.settings) { openSettings() }]
    ],
    destructiveActions: [                              // Red items at bottom
        MenuAction(title: "Delete", icon: AppIcon.delete) { delete() }
    ]
) {
    Label("More", systemImage: AppIcon.more)           // Menu trigger label
}
```

### Toolbars

**TopLevelToolbarContent** — For top-level views (Home, Books, Threads, Search). Shows avatar + optional overflow menu.

```swift
.toolbar {
    TopLevelToolbarContent(
        avatarData: userProfile.avatarImageData,
        avatarInitials: userProfile.initials,
        onAvatarTap: { navManager.presentSheet(.settings) },
        listSections: [[
            MenuAction(title: "Import ePub", icon: AppIcon.importBook) { showImporter = true }
        ]]
    )
}
```

**DetailToolbarContent** — For detail views. Single action shown directly, overflow for extras.

```swift
.toolbar {
    DetailToolbarContent(
        primaryAction: MenuAction(title: "Settings", icon: AppIcon.settings) {
            showSettings = true
        }
    )
}
```

### Navigation

**AppDestination** — Universal navigation enum. Use instead of raw `PersistentIdentifier`.

```swift
NavigationLink(value: AppDestination.book(book.persistentModelID)) {
    LargeCard(...)
}
.matchedTransitionSource(id: book.persistentModelID, in: namespace)
```

**AppNavigationDestinations** — Apply to each `NavigationStack` root. Provides `@Namespace`, injects `\.appNamespace`, and registers `.navigationDestination(for: AppDestination.self)` with zoom transitions.

```swift
NavigationStack(path: $path) {
    MyListView()
        .appNavigationDestinations()
}
```
