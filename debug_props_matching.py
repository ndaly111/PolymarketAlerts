#!/usr/bin/env python3
"""
Debug script to diagnose props matching issues.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from kalshi_props import fetch_kalshi_props, KALSHI_TO_ODDSAPI
from oddsapi_props import fetch_all_props
from props_value import match_props, filter_and_rank


def main():
    print("=" * 70)
    print("PROPS MATCHING DEBUG")
    print("=" * 70)

    # Fetch Kalshi props
    print("\n[1] Fetching Kalshi props...")
    kalshi_props = fetch_kalshi_props()
    print(f"    Found {len(kalshi_props)} Kalshi props")

    # Show Kalshi stat type breakdown
    kalshi_types = Counter(p.get("stat_type") for p in kalshi_props)
    print(f"\n    Kalshi stat types:")
    for stat_type, count in kalshi_types.most_common():
        oddsapi_type = KALSHI_TO_ODDSAPI.get(stat_type, "UNMAPPED")
        print(f"      {stat_type}: {count} -> {oddsapi_type}")

    # Sample Kalshi props
    print(f"\n    Sample Kalshi props (first 5):")
    for p in kalshi_props[:5]:
        print(f"      {p['player_name_norm'][:20]:<20} | {p['oddsapi_type']:<20} | line={p['line']}")

    # Fetch Odds API props
    print("\n[2] Fetching Odds API props...")
    odds_props = fetch_all_props(min_books=2)
    print(f"    Found {len(odds_props)} Odds API props")

    # Show Odds API prop type breakdown
    odds_types = Counter(p.get("prop_type") for p in odds_props)
    print(f"\n    Odds API prop types:")
    for prop_type, count in odds_types.most_common():
        print(f"      {prop_type}: {count}")

    # Sample Odds API props
    print(f"\n    Sample Odds API props (first 5):")
    for p in odds_props[:5]:
        print(f"      {p['player_name_norm'][:20]:<20} | {p['prop_type']:<20} | line={p['line']}")

    # Build lookup sets for debugging
    kalshi_keys = set()
    for p in kalshi_props:
        player_norm = p.get("player_name_norm", "")
        oddsapi_type = p.get("oddsapi_type", "")
        if player_norm and oddsapi_type:
            kalshi_keys.add((player_norm, oddsapi_type))

    odds_keys = set()
    for p in odds_props:
        player_norm = p.get("player_name_norm", "")
        prop_type = p.get("prop_type", "")
        if player_norm and prop_type:
            odds_keys.add((player_norm, prop_type))

    # Find overlapping player+type combinations
    overlap = kalshi_keys & odds_keys
    print(f"\n[3] Key overlap analysis:")
    print(f"    Kalshi unique (player, type): {len(kalshi_keys)}")
    print(f"    Odds API unique (player, type): {len(odds_keys)}")
    print(f"    Overlap (player+type match): {len(overlap)}")

    if overlap:
        print(f"\n    Sample overlapping keys (first 10):")
        for key in list(overlap)[:10]:
            print(f"      {key}")

    # Check for player name issues
    kalshi_players = set(p.get("player_name_norm") for p in kalshi_props if p.get("player_name_norm"))
    odds_players = set(p.get("player_name_norm") for p in odds_props if p.get("player_name_norm"))
    player_overlap = kalshi_players & odds_players

    print(f"\n[4] Player name analysis:")
    print(f"    Kalshi unique players: {len(kalshi_players)}")
    print(f"    Odds API unique players: {len(odds_players)}")
    print(f"    Player name overlap: {len(player_overlap)}")

    # Sample players only in Kalshi
    kalshi_only = kalshi_players - odds_players
    if kalshi_only:
        print(f"\n    Sample players ONLY in Kalshi (first 10):")
        for name in list(kalshi_only)[:10]:
            print(f"      '{name}'")

    # Sample players only in Odds API
    odds_only = odds_players - kalshi_players
    if odds_only:
        print(f"\n    Sample players ONLY in Odds API (first 10):")
        for name in list(odds_only)[:10]:
            print(f"      '{name}'")

    # Run actual matching
    print(f"\n[5] Running match_props()...")
    matches = match_props(odds_props, kalshi_props)
    print(f"    Total matches (before edge filter): {len(matches)}")

    if matches:
        # Show edge distribution
        edges = [m.get("net_edge", m.get("best_edge", 0)) for m in matches]
        positive_edges = [e for e in edges if e > 0]
        negative_edges = [e for e in edges if e < 0]

        print(f"\n    Edge distribution:")
        print(f"      Positive edges: {len(positive_edges)}")
        print(f"      Negative edges: {len(negative_edges)}")
        if edges:
            print(f"      Min edge: {min(edges):.1%}")
            print(f"      Max edge: {max(edges):.1%}")
            print(f"      Avg edge: {sum(edges)/len(edges):.1%}")

        # Count by edge buckets
        buckets = Counter()
        for e in edges:
            if e < 0:
                buckets["< 0%"] += 1
            elif e < 0.02:
                buckets["0-2%"] += 1
            elif e < 0.05:
                buckets["2-5%"] += 1
            elif e < 0.10:
                buckets["5-10%"] += 1
            else:
                buckets["10%+"] += 1

        print(f"\n    Edge buckets:")
        for bucket in ["< 0%", "0-2%", "2-5%", "5-10%", "10%+"]:
            print(f"      {bucket}: {buckets.get(bucket, 0)}")

        # Show sample matches
        print(f"\n    Sample matches (sorted by edge, first 10):")
        sorted_matches = sorted(matches, key=lambda x: x.get("net_edge", x.get("best_edge", 0)), reverse=True)
        for m in sorted_matches[:10]:
            player = m.get("player_name", "")[:15]
            stat = m.get("stat_type", "")[:6]
            line = m.get("kalshi_line", 0)
            side = m.get("best_side", "")
            edge = m.get("net_edge", m.get("best_edge", 0))
            kalshi_prob = m.get("kalshi_over_prob", 0) if side == "OVER" else m.get("kalshi_under_prob", 0)
            fair_prob = m.get("odds_over_prob", 0) if side == "OVER" else m.get("odds_under_prob", 0)
            books = m.get("odds_books_used", 0)

            print(f"      {player:<15} {stat:<6} {line:>4.0f}+ {side:<5} | edge={edge:>+6.1%} | kalshi={kalshi_prob:.1%} fair={fair_prob:.1%} | books={books}")

    # Filter and rank
    print(f"\n[6] Running filter_and_rank(min_edge=0.05)...")
    opportunities = filter_and_rank(matches, min_edge=0.05, top_n=50)
    print(f"    Opportunities with 5%+ edge: {len(opportunities)}")

    if opportunities:
        print(f"\n    Top opportunities:")
        for opp in opportunities[:5]:
            player = opp.get("player_name", "")
            stat = opp.get("stat_type", "")
            line = opp.get("kalshi_line", 0)
            side = opp.get("best_side", "")
            edge = opp.get("net_edge", opp.get("best_edge", 0))
            print(f"      {player} {stat} {line:.0f}+ {side} | edge={edge:+.1%}")

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
