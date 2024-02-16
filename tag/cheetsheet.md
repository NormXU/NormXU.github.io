---
layout: tag
title: "Tag: CheetSheet"
permalink: /t/cheatsheet
---
This page is a repository for cheatsheet collecting for a quick reference.
<ul class="post-list">
  {%- for post in site.tags["CheetSheet"] -%}
    <li>
      {%- assign date_format = site.minima.date_format | default: "%Y-%m-%d" -%}
      <span class="post-meta">
        {{ post.date | date: date_format }}
      </span>
      <a class="post-link" href="{{ post.url | relative_url }}">
          {{ post.title | escape }}
      </a>
    </li>
  {%- endfor -%}
</ul>