//
//  CommunityBoardView 2.swift
//  InSite
//
//  Created by Anand Parikh on 9/29/25.
//


struct CommunityBoardView_Previews: PreviewProvider {
    static var previews: some View {
        Group {
            NavigationStack { CommunityBoardView(accent: .blue) }
                .environmentObject(ThemeManager())
                .environment(\.colorScheme, .light)

            NavigationStack { CommunityBoardView(accent: .pink) }
                .environmentObject(ThemeManager())
                .environment(\.colorScheme, .dark)
        }
    }
}
